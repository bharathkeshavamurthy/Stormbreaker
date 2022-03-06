//----------------------------------------------------------------------
//-- Downsize an AXI stream to half the data width
//----------------------------------------------------------------------

`default_nettype none

module stream_dnsizer # (
    parameter IN_DATA_WIDTH = 32,

    parameter USER_WIDTH = 1,

    //0 = first transfer comes from the low half
    //1 = first transfer comes from the high half
    parameter HIGH_FIRST = 1'b0)
(
    input wire clk,
    input wire rst,

    //input bus at DATA_WIDTH with tlast
    input wire [IN_DATA_WIDTH-1:0] in_tdata,
    input wire [USER_WIDTH-1:0] in_tuser,
    input wire [(IN_DATA_WIDTH/8)-1:0] in_tkeep,
    input wire in_tlast,
    input wire in_tvalid,
    output wire in_tready,

    //output bus at 1/2 DATA_WIDTH with tlast
    output wire [(IN_DATA_WIDTH/2)-1:0] out_tdata,
    output wire [USER_WIDTH-1:0] out_tuser,
    output wire [(IN_DATA_WIDTH/16)-1:0] out_tkeep,
    output wire out_tlast,
    output wire out_tvalid,
    input wire out_tready
);

    wire [IN_DATA_WIDTH-1:0] full_tdata;
    wire [USER_WIDTH-1:0] full_tuser;
    wire [(IN_DATA_WIDTH/8)-1:0] full_tkeep;
    wire full_tlast;
    wire full_tvalid;
    wire full_tready;

    stream_fifo_srl32 #(.DATA_WIDTH(USER_WIDTH + (IN_DATA_WIDTH/8) + IN_DATA_WIDTH)) in_fifo
    (
        .clk(clk),
        .rst(rst),
        .in_tdata({in_tuser, ((in_tkeep == 0)?~in_tkeep:in_tkeep), in_tdata}),
        .in_tlast(in_tlast),
        .in_tvalid(in_tvalid),
        .in_tready(in_tready),
        .out_tdata({full_tuser, full_tkeep, full_tdata}),
        .out_tlast(full_tlast),
        .out_tvalid(full_tvalid),
        .out_tready(full_tready)
    );

    wire [(IN_DATA_WIDTH/2)-1:0] half_tdata;
    wire [USER_WIDTH-1:0] half_tuser;
    wire [(IN_DATA_WIDTH/16)-1:0] half_tkeep;
    wire half_tlast;
    wire half_tvalid;
    wire half_tready;

    assign out_tdata = half_tdata;
    assign out_tuser = half_tuser;
    assign out_tkeep = half_tkeep;
    assign out_tlast = half_tlast;
    assign out_tvalid = half_tvalid;
    assign half_tready = out_tready;

    reg phase;
    always @(posedge clk) begin
        if (rst) phase = 1'b0;
        else if (half_tvalid && half_tready) phase = half_tlast?1'b0:!phase;
    end

    assign half_tdata = (phase ^ HIGH_FIRST)?
        full_tdata[IN_DATA_WIDTH-1:(IN_DATA_WIDTH/2)]:
        full_tdata[(IN_DATA_WIDTH/2)-1:0];

    //same tuser for both output cycles
    assign half_tuser = full_tuser;

    assign half_tkeep = (phase ^ HIGH_FIRST)?
        full_tkeep[(IN_DATA_WIDTH/8)-1:(IN_DATA_WIDTH/16)]:
        full_tkeep[(IN_DATA_WIDTH/16)-1:0];

    wire [(IN_DATA_WIDTH/16)-1:0] next_half_tkeep = (phase ^ HIGH_FIRST)?
        full_tkeep[(IN_DATA_WIDTH/16)-1:0]:
        full_tkeep[(IN_DATA_WIDTH/8)-1:(IN_DATA_WIDTH/16)];

    //allow early tlast when tkeep is enabled but second half says not to keep
    assign half_tlast = (!phase && (full_tkeep != 0) && (next_half_tkeep == 0))?full_tlast:
        (full_tlast && phase); //only tlast on second phase

    assign half_tvalid = full_tvalid; //valid for both phases

    //only read on second phase unless early tlast
    //ready hold-off when we can actually take data is bad
    //interface protocol, but internal in_fifo protects us
    assign full_tready = half_tready && (phase || half_tlast);

endmodule //stream_dnsizer

`default_nettype wire
