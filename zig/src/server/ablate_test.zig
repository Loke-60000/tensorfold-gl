//! TENSORFOLD_ABLATE_MODEL: a request naming that id reaches the engine ablated, any other id does not.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const openai = @import("openai.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const ByteText = @import("embedding_protocol_test.zig").ByteText;

const Capture = struct {
    ablate: ?bool = null,

    fn engine(e: *Capture) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn info(_: *anyopaque) api.Info {
        return .{ .ablate = true };
    }

    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *Capture = @ptrCast(@alignCast(ctx));
        e.ablate = request.ablate;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = &.{ 'o', 'k' } });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }

    fn cancel(_: *anyopaque, _: api.Id) void {}

    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }

    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

const Reply = struct {
    status: ?u16 = null,

    fn out(r: *Reply) openai.Out {
        return .{ .ctx = r, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }

    fn open(_: *anyopaque, _: usize) error{Closed}!void {}

    fn event(_: *anyopaque, _: ?json.Value) error{Closed}!void {}

    fn reply(ctx: *anyopaque, status: u16, _: json.Value) void {
        const r: *Reply = @ptrCast(@alignCast(ctx));
        r.status = status;
    }
};

/// Whether a completion for `model` reached the engine ablated, on a server whose ablation model is `ablate_model`.
fn ablated(model: []const u8, ablate_model: ?[]const u8) !?bool {
    var backend: Capture = .{};
    const srv = try Server.init(std.testing.allocator, std.testing.io, backend.engine(), ByteText.model(), .{
        .served_name = "m",
        .model_ids = &.{ "m", "m-ablated" },
        .ablate_model = ablate_model,
        .enable_thinking = false,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const raw = (try json.parse(a, try std.fmt.allocPrint(a, "{{\"model\":\"{s}\",\"max_tokens\":2,\"prompt\":[104,105]}}", .{model}))).ok;
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var reply: Reply = .{};
    openai.run(srv, a, reply.out(), .{ .conn = &conn }, false, raw);
    try std.testing.expectEqual(@as(?u16, 200), reply.status);
    return backend.ablate;
}

test "ablation model: only its requests are ablated" {
    try std.testing.expectEqual(@as(?bool, true), try ablated("m-ablated", "m-ablated"));
    try std.testing.expectEqual(@as(?bool, false), try ablated("m", "m-ablated"));
    try std.testing.expectEqual(@as(?bool, false), try ablated("m-ablated", null));
}
