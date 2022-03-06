//----------------------------------------------------------------------
//-- Combine two axi streams with tlast into one stream
//--
//-- One of the streams always takes priority when both inputs are ready.
//-- Stream flow only switches after an entire packet transfer completes.
//-- tuser signal is provided as a stream ID to differentiate streams.
//----------------------------------------------------------------------

`default_nettype none

module stream_combine_prio # (
    parameter USER_WIDTH = 1,
    parameter DATA_WIDTH = 32,
    parameter KEEP_WIDTH = DATA_WIDTH/8)
(
    input wire clk,
    input wire rst,

    input wire [DATA_WIDTH-1:0] in_tdata,
    input wire [KEEP_WIDTH-1:0] in_tkeep,
    input wire [USER_WIDTH-1:0] in_tuser,
    input wire in_tlast,
    input wire in_tvalid,
    output wire in_tready,

    input wire [DATA_WIDTH-1:0] prio_tdata,
    input wire [KEEP_WIDTH-1:0] prio_tkeep,
    input wire [USER_WIDTH-1:0] prio_tuser,
    input wire prio_tlast,
    input wire prio_tvalid,
    output wire prio_tready,

    output wire [0:0] out_tdest,
    output wire [DATA_WIDTH-1:0] out_tdata,
    output wire [KEEP_WIDTH-1:0] out_tkeep,
    output wire [USER_WIDTH-1:0] out_tuser,
    output wire out_tlast,
    output wire out_tvalid,
    input wire out_tready
);

    reg [0:0] state;
    reg prio;
    localparam ST_WAIT_RDY = 0;
    localparam ST_FWD_PKT = 1;

    always @(posedge clk) begin
        if (rst) begin
            state <= ST_WAIT_RDY;
            prio <= 1'b0;
        end
        else if (state == ST_WAIT_RDY) begin
            if (prio_tvalid || in_tvalid) state <= ST_FWD_PKT;
            prio <= prio_tvalid;
        end
        else if (state == ST_FWD_PKT) begin
            if (out_tvalid && out_tready && out_tlast) state <= ST_WAIT_RDY;
        end
    end

    assign out_tdest = prio; //prio is 1, otherwise 0
    assign out_tdata = prio?prio_tdata:in_tdata;
    assign out_tkeep = prio?prio_tkeep:in_tkeep;
    assign out_tuser = prio?prio_tuser:in_tuser;
    assign out_tlast = prio?prio_tlast:in_tlast;
    assign out_tvalid = (prio?prio_tvalid:in_tvalid) && (state == ST_FWD_PKT);
    assign in_tready = !prio && out_tready && (state == ST_FWD_PKT);
    assign prio_tready = prio && out_tready && (state == ST_FWD_PKT);

endmodule //stream_combine_prio

`default_nettype wire
