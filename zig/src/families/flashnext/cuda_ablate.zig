//! Per-request abliteration: an abliterated build edits the residual-stream writers, W' = W - R diag(lam) R^T W, so its
//! output is y - R diag(lam) (R^T y). The engine applies that correction to the writers' outputs at run time, for the rows
//! whose request named TENSORFOLD_ABLATE_MODEL, before the hyper-connection write-back: the token mixers' output
//! projections (linear_attn.out_proj, self_attn.o_proj; their fp32 K slices in the batched pass) with lam_attn and the
//! shared expert's down projection (its own slot of the MoE output) with lam_shared, from `first_layer` up. The MTP head
//! is untouched (the main forward verifies its drafts).
//!
//! TENSORFOLD_ABLATE names a safetensors file: `directions` [k, d] fp32 (orthonormal rows), `lam_attn` [k] and
//! `lam_shared` [k] fp32, `first_layer` [1] int64. TENSORFOLD_ABLATE_SCALE multiplies both strengths (default 1).
//! With TENSORFOLD_ABLATE unset nothing here runs and the forward is unchanged.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");

pub const max_k = 4;

pub const Ablation = struct {
    /// [k * d directions | k lam_attn | k lam_shared] fp32
    dev: cuda.DeviceBuffer,
    k: usize,
    d: usize,
    first: usize,

    pub const Kind = enum { attn, shared };

    /// The directions file TENSORFOLD_ABLATE names, on the device; null when it is unset.
    pub fn fromEnv(gpa: std.mem.Allocator, io: std.Io, drv: *const cuda.Driver, hidden: usize, world: usize) !?Ablation {
        const path = env("TENSORFOLD_ABLATE") orelse return null;
        if (world != 1) return error.AblateNeedsOneRank;
        const scale: f32 = if (env("TENSORFOLD_ABLATE_SCALE")) |v| std.fmt.parseFloat(f32, v) catch return error.BadAblateScale else 1;
        var file = try core.safetensors.File.open(gpa, io, path);
        defer file.close(io);
        const dirs = file.get("directions") orelse return error.AblateFileMissesDirections;
        if (dirs.dtype != .f32 or dirs.rank != 2 or dirs.shape[1] != hidden) return error.AblateDirectionsShape;
        const k = dirs.shape[0];
        if (k < 1 or k > max_k) return error.AblateDirectionsShape;
        const lam_attn = file.get("lam_attn") orelse return error.AblateFileMissesLambdas;
        const lam_shared = file.get("lam_shared") orelse return error.AblateFileMissesLambdas;
        if (!lam_attn.is(.f32, &.{k}) or !lam_shared.is(.f32, &.{k})) return error.AblateLambdaShape;
        const first_t = file.get("first_layer") orelse return error.AblateFileMissesFirstLayer;
        if (!first_t.is(.i64, &.{1})) return error.AblateFirstLayerShape;
        const first = std.mem.readInt(i64, first_t.bytes[0..8], .little);
        if (first < 0) return error.AblateFirstLayerShape;

        const host = try gpa.alloc(f32, k * hidden + 2 * k);
        defer gpa.free(host);
        // the mapped bytes may be unaligned: copied out before they are read as floats
        @memcpy(std.mem.sliceAsBytes(host[0 .. k * hidden]), dirs.bytes);
        @memcpy(std.mem.sliceAsBytes(host[k * hidden ..][0..k]), lam_attn.bytes);
        @memcpy(std.mem.sliceAsBytes(host[k * hidden + k ..][0..k]), lam_shared.bytes);
        for (0..k) |a| for (0..k) |b| {
            var dot: f64 = 0;
            for (host[a * hidden ..][0..hidden], host[b * hidden ..][0..hidden]) |x, y| dot += @as(f64, x) * y;
            const want: f64 = if (a == b) 1 else 0;
            if (@abs(dot - want) > 1e-3) return error.AblateDirectionsNotOrthonormal;
        };
        for (host[k * hidden ..]) |*l| l.* *= scale;

        var dev = try cuda.DeviceBuffer.alloc(drv, host.len * 4);
        errdefer dev.free();
        try dev.upload(0, std.mem.sliceAsBytes(host));
        const la = host[k * hidden ..][0..k];
        const ls = host[k * hidden + k ..][0..k];
        std.log.info("ablation: {s}, {d} directions from layer {d}, lam_attn {any}, lam_shared {any} (scale {d})", .{ path, k, first, la, ls, scale });
        return .{ .dev = dev, .k = k, .d = hidden, .first = @intCast(first) };
    }

    pub fn deinit(a: *Ablation) void {
        a.dev.free();
        a.* = undefined;
    }

    /// The correction on one writer's output: `rows` fp32 rows `row_stride` elements apart, in `slices` slices
    /// `slice_stride` apart (their sum is the output; slice 0 takes the correction), for the rows `mask` [rows] fp32 flags.
    /// Layers before `first` are left as they are.
    pub fn apply(a: *const Ablation, f: cuda.Function, stream: cuda.Stream, layer: usize, kind: Kind, y: u64, row_stride: usize, slice_stride: usize, slices: usize, mask: u64, rows: usize) !void {
        if (layer < a.first or rows == 0) return;
        if (rows > 65535 or slices < 1 or row_stride < a.d) return error.AblateRows;
        const dirs = a.dev.ptr;
        const lam = dirs + (a.k * a.d + @as(usize, if (kind == .attn) 0 else a.k)) * 4;
        var args: cuda.Args = .{};
        args.add(y);
        args.add(@as(c_longlong, @intCast(row_stride)));
        args.add(@as(c_longlong, @intCast(slice_stride)));
        args.add(@as(c_int, @intCast(slices)));
        args.add(mask);
        args.add(dirs);
        args.add(lam);
        args.add(@as(c_int, @intCast(a.d)));
        args.add(@as(c_int, @intCast(a.k)));
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, stream, &args);
    }
};

fn env(name: [:0]const u8) ?[]const u8 {
    return if (std.c.getenv(name)) |v| std.mem.span(v) else null;
}
