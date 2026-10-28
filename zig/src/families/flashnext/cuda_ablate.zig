//! Per-request abliteration, an identity a model id: an abliterated build edits the residual-stream writers,
//! W' = W - R diag(lam) R^T W, so its output is y - R diag(lam) (R^T y). The engine applies that correction to the
//! writers' outputs at run time, for the rows whose request named an identity (1..n in the list's order; 0 is the
//! stock model), before the hyper-connection write-back: the token mixers' output projections (linear_attn.out_proj,
//! self_attn.o_proj; their fp32 K slices in the batched pass) with lam_attn and the shared expert's down projection
//! (its own slot of the MoE output) with lam_shared, from the identity's `first_layer` up. The MTP head is untouched
//! (the main forward verifies its drafts).
//!
//! TENSORFOLD_IDENTITIES lists the identities (core/identities.zig), each a safetensors file: `directions` [k, d] fp32
//! (orthonormal rows), `lam_attn` [k] and `lam_shared` [k] fp32, `first_layer` [1] int64. TENSORFOLD_ABLATE_SCALE
//! multiplies every strength (default 1). With no identity listed nothing here runs and the forward is unchanged.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");

pub const max_k = 4;

/// Floats an identity takes in the table: max_k directions, then max_k lam_attn and max_k lam_shared (k of each used).
fn strideOf(d: usize) usize {
    return max_k * d + 2 * max_k;
}

pub const Ablation = struct {
    /// [n * stride] fp32: each identity's [k * d directions | k lam_attn | k lam_shared], packed from its block's start
    table: cuda.DeviceBuffer,
    /// [n * 2] i32: each identity's k and first layer
    meta: cuda.DeviceBuffer,
    n: usize,
    d: usize,
    /// The lowest first layer of them: the launch is skipped below it.
    first: usize,

    pub const Kind = enum(u8) { attn = 0, shared = 1 };

    /// The listed identities' files on the device; null when none is listed.
    pub fn fromEnv(gpa: std.mem.Allocator, io: std.Io, drv: *const cuda.Driver, hidden: usize, world: usize) !?Ablation {
        var names: [core.identities.max]core.identities.Identity = undefined;
        const listed = try core.identities.list(env("TENSORFOLD_IDENTITIES"), env("TENSORFOLD_ABLATE"), env("TENSORFOLD_ABLATE_MODEL"), &names);
        if (listed.len == 0) return null;
        if (world != 1) return error.AblateNeedsOneRank;
        const scale: f32 = if (env("TENSORFOLD_ABLATE_SCALE")) |v| std.fmt.parseFloat(f32, v) catch return error.BadAblateScale else 1;
        const stride = strideOf(hidden);

        const host = try gpa.alloc(f32, listed.len * stride);
        defer gpa.free(host);
        @memset(host, 0);
        const meta = try gpa.alloc(i32, listed.len * 2);
        defer gpa.free(meta);
        var first: usize = std.math.maxInt(usize);
        for (listed, 0..) |ident, i| {
            const loaded = try load(gpa, io, ident.path, hidden, scale, host[i * stride ..][0..stride]);
            meta[2 * i] = @intCast(loaded.k);
            meta[2 * i + 1] = @intCast(loaded.first);
            first = @min(first, loaded.first);
            const block = host[i * stride ..];
            std.log.info("identity {d} {s}: {s}, {d} directions from layer {d}, lam_attn {any}, lam_shared {any} (scale {d})", .{ i + 1, ident.id, ident.path, loaded.k, loaded.first, block[loaded.k * hidden ..][0..loaded.k], block[loaded.k * hidden + loaded.k ..][0..loaded.k], scale });
        }

        var table = try cuda.DeviceBuffer.alloc(drv, host.len * 4);
        errdefer table.free();
        try table.upload(0, std.mem.sliceAsBytes(host));
        var meta_dev = try cuda.DeviceBuffer.alloc(drv, meta.len * 4);
        errdefer meta_dev.free();
        try meta_dev.upload(0, std.mem.sliceAsBytes(meta));
        return .{ .table = table, .meta = meta_dev, .n = listed.len, .d = hidden, .first = first };
    }

    const Loaded = struct { k: usize, first: usize };

    /// One identity's file into its block: directions, then lam_attn and lam_shared times `scale`.
    fn load(gpa: std.mem.Allocator, io: std.Io, path: []const u8, hidden: usize, scale: f32, block: []f32) !Loaded {
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

        // the mapped bytes may be unaligned: copied out before they are read as floats
        @memcpy(std.mem.sliceAsBytes(block[0 .. k * hidden]), dirs.bytes);
        @memcpy(std.mem.sliceAsBytes(block[k * hidden ..][0..k]), lam_attn.bytes);
        @memcpy(std.mem.sliceAsBytes(block[k * hidden + k ..][0..k]), lam_shared.bytes);
        for (0..k) |a| for (0..k) |b| {
            var dot: f64 = 0;
            for (block[a * hidden ..][0..hidden], block[b * hidden ..][0..hidden]) |x, y| dot += @as(f64, x) * y;
            const want: f64 = if (a == b) 1 else 0;
            if (@abs(dot - want) > 1e-3) return error.AblateDirectionsNotOrthonormal;
        };
        for (block[k * hidden ..][0 .. 2 * k]) |*l| l.* *= scale;
        return .{ .k = k, .first = @intCast(first) };
    }

    pub fn deinit(a: *Ablation) void {
        a.table.free();
        a.meta.free();
        a.* = undefined;
    }

    /// The correction on one writer's output: `rows` fp32 rows `row_stride` elements apart, in `slices` slices
    /// `slice_stride` apart (their sum is the output; slice 0 takes the correction), each row's identity in `who`
    /// [rows] fp32 (0: left alone). Layers before every identity's first are left as they are.
    pub fn apply(a: *const Ablation, f: cuda.Function, stream: cuda.Stream, layer: usize, kind: Kind, y: u64, row_stride: usize, slice_stride: usize, slices: usize, who: u64, rows: usize) !void {
        if (layer < a.first or rows == 0) return;
        if (rows > 65535 or slices < 1 or row_stride < a.d) return error.AblateRows;
        var args: cuda.Args = .{};
        args.add(y);
        args.add(@as(c_longlong, @intCast(row_stride)));
        args.add(@as(c_longlong, @intCast(slice_stride)));
        args.add(@as(c_int, @intCast(slices)));
        args.add(who);
        args.add(a.table.ptr);
        args.add(a.meta.ptr);
        args.add(@as(c_int, @intCast(strideOf(a.d))));
        args.add(@as(c_int, @intFromEnum(kind)));
        args.add(@as(c_int, @intCast(layer)));
        args.add(@as(c_int, @intCast(a.d)));
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, stream, &args);
    }
};

fn env(name: [:0]const u8) ?[]const u8 {
    return if (std.c.getenv(name)) |v| std.mem.span(v) else null;
}
