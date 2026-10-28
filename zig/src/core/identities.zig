//! The identities a server offers beside the stock model: a model id each, with the directions file its requests get
//! (families/flashnext/cuda_ablate.zig). TENSORFOLD_IDENTITIES lists them as `id=path` entries separated by commas or
//! newlines, in the order the engine numbers them (1..n; 0 is the stock model). TENSORFOLD_ABLATE with
//! TENSORFOLD_ABLATE_MODEL names one the old way and stays the whole list while TENSORFOLD_IDENTITIES is unset.
const std = @import("std");

pub const max = 32;

pub const Identity = struct { id: []const u8, path: []const u8 };

pub const Error = error{ TooMany, Malformed, Duplicate };

/// The listed identities in order, in `out`; `legacy_path` and `legacy_id` are TENSORFOLD_ABLATE and TENSORFOLD_ABLATE_MODEL.
pub fn list(spec: ?[]const u8, legacy_path: ?[]const u8, legacy_id: ?[]const u8, out: *[max]Identity) Error![]Identity {
    var n: usize = 0;
    if (spec) |s| {
        var it = std.mem.tokenizeAny(u8, s, ",\n");
        while (it.next()) |raw| {
            const entry = std.mem.trim(u8, raw, " \t\r");
            if (entry.len == 0) continue;
            const eq = std.mem.indexOfScalar(u8, entry, '=') orelse return error.Malformed;
            const id = std.mem.trim(u8, entry[0..eq], " \t");
            const path = std.mem.trim(u8, entry[eq + 1 ..], " \t");
            if (id.len == 0 or path.len == 0) return error.Malformed;
            for (out[0..n]) |seen| if (std.mem.eql(u8, seen.id, id)) return error.Duplicate;
            if (n == max) return error.TooMany;
            out[n] = .{ .id = id, .path = path };
            n += 1;
        }
    } else if (legacy_path) |p| if (legacy_id) |id| {
        const path = std.mem.trim(u8, p, " \t\r\n");
        const name = std.mem.trim(u8, id, " \t\r\n");
        if (path.len > 0 and name.len > 0) {
            out[0] = .{ .id = name, .path = path };
            n = 1;
        }
    };
    return out[0..n];
}

test "identities: the list in order, the legacy pair, the errors" {
    var out: [max]Identity = undefined;
    const two = try list(" skuld = /a/skuld.safetensors ,\nmuse=/b/muse.safetensors,", null, null, &out);
    try std.testing.expectEqual(@as(usize, 2), two.len);
    try std.testing.expectEqualStrings("skuld", two[0].id);
    try std.testing.expectEqualStrings("/a/skuld.safetensors", two[0].path);
    try std.testing.expectEqualStrings("muse", two[1].id);
    try std.testing.expectEqualStrings("/b/muse.safetensors", two[1].path);

    const legacy = try list(null, "/a/skuld.safetensors", "gl-zero-skuld", &out);
    try std.testing.expectEqual(@as(usize, 1), legacy.len);
    try std.testing.expectEqualStrings("gl-zero-skuld", legacy[0].id);
    try std.testing.expectEqual(@as(usize, 0), (try list(null, "/a/skuld.safetensors", null, &out)).len);
    // the list set, the legacy pair is ignored
    try std.testing.expectEqual(@as(usize, 0), (try list("", "/a/skuld.safetensors", "gl-zero-skuld", &out)).len);

    try std.testing.expectError(error.Malformed, list("skuld", null, null, &out));
    try std.testing.expectError(error.Malformed, list("=/a", null, null, &out));
    try std.testing.expectError(error.Duplicate, list("a=/1,a=/2", null, null, &out));
    var long: [(max + 1) * 12]u8 = undefined;
    var at: usize = 0;
    for (0..max + 1) |i| at += (try std.fmt.bufPrint(long[at..], "i{d}=/p{d},", .{ i, i })).len;
    try std.testing.expectError(error.TooMany, list(long[0..at], null, null, &out));
}
