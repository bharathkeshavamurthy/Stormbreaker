//----------------------------------------------------------------------
//-- Upsize an AXI stream to twice the data width
//----------------------------------------------------------------------

`default_nettype none

module stream_upsizer # (
    parameter IN_DATA_WIDTH = 32,

    parameter USER_WIDTH = 1,

    //0 = low half comes from first input transfer
    //1 = high half comes from first input transfer
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

    //output bus at 2x DATA_WIDTH with tlast
    output wire [(IN_DATA_WIDTH*2)-1:0] out_tdata,
    output wire [USER_WIDTH-1:0] out_tuser,
    output wire [(IN_DATA_WIDTH/4)-1:0] out_tkeep,
    output wire out_tlast,
    output wire out_tvalid,
    input wire out_tready
);

    wire [IN_DATA_WIDTH-1:0] half_tdata;
    wire [USER_WIDTH-1:0] half_tuser;
    wire [(IN_DATA_WIDTH/8)-1:0] half_tkeep;
    wire half_tlast;
    wire half_tvalid;
    wire half_tready;

    assign half_tdata = in_tdata;
    assign half_tuser = in_tuser;
    assign half_tkeep = (in_tkeep == 0)?~in_tkeep:in_tkeep;
    assign half_tlast = in_tlast;
    assign half_tvalid = in_tvalid;
    assign in_tready = half_tready;

    wire [(IN_DATA_WIDTH*2)-1:0] full_tdata;
    wire [USER_WIDTH-1:0] full_tuser;
    wire [(IN_DATA_WIDTH/4)-1:0] full_tkeep;
    wire full_tlast;
    wire full_tvalid;
    wire full_tready;

    stream_fifo_srl32 #(.DATA_WIDTH(USER_WIDTH + (IN_DATA_WIDTH/4) + (IN_DATA_WIDTH*2))) out_fifo
    (
        .clk(clk),
        .rst(rst),
        .in_tdata({full_tuser, full_tkeep, full_tdata}),
        .in_tlast(full_tlast),
        .in_tvalid(full_tvalid),
        .in_tready(full_tready),
        .out_tdata({out_tuser, out_tkeep, out_tdata}),
        .out_tlast(out_tlast),
        .out_tvalid(out_tvalid),
        .out_tready(out_tready)
    );

    reg [IN_DATA_WIDTH-1:0] prev_tdata;
    reg [USER_WIDTH-1:0] prev_tuser;
    reg [(IN_DATA_WIDTH/8)-1:0] prev_tkeep;
    reg phase;
    always @(posedge clk) begin
        if (rst) begin
            phase <= 1'b0;
            prev_tdata <= 0;
            prev_tuser <= 0;
            prev_tkeep <= 0;
        end
        else if (half_tvalid && half_tready) begin
            phase <= full_tvalid?1'b0:1'b1;
            prev_tdata <= full_tvalid?0:half_tdata;
            prev_tuser <= full_tvalid?0:half_tuser;
            prev_tkeep <= full_tvalid?0:half_tkeep;
        end
    end

    //mux data based on msb and phase (phase supports early tlast)
    assign full_tdata = (phase ^ HIGH_FIRST)?
        {half_tdata, prev_tdata}:
        {prev_tdata, half_tdata};

    //use the previous tuser unless this is an early last
    assign full_tuser = phase?prev_tuser:half_tuser;

    assign full_tkeep = (phase ^ HIGH_FIRST)?
        {half_tkeep, prev_tkeep}:
        {prev_tkeep, half_tkeep};

    assign full_tlast = half_tlast;

    assign full_tvalid = half_tvalid && (phase || half_tlast); //valid for early tlast

    //only read on second phase unless early tlast
    //ready hold-off when we can actually take data is bad
    //interface protocol, but internal in_fifo protects us
    assign half_tready = full_tready;

endmodule //stream_upsizer

`default_nettype wire
