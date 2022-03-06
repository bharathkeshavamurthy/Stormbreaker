//----------------------------------------------------------------------
//-- Clock domain crossing stream fifo with arbitrary width
//--
//-- The fifo depth is a fixed value based on the width.
//-- 16kb for 32-bit streams, and 32kb for 64-bit streams.
//-- Streams larger than 64-bits use multiple parallel FIFOs.
//----------------------------------------------------------------------

`default_nettype none

module cdc_stream_fifo # (
    parameter FIFO_MEMORY_TYPE = "block",
    parameter DATA_WIDTH = 32,
    parameter FIFO_DEPTH = 512, //single hard fifo by default
    parameter DATA_COUNT_WIDTH = $clog2(FIFO_DEPTH)+1,
    parameter ENABLE_TKEEP = 0, //set to 1 to use tkeep signals
    parameter ENABLE_TUSER = 0, //set to 1 to use tuser signals
    parameter USER_WIDTH = 1,
    parameter KEEP_WIDTH = ENABLE_TKEEP?(DATA_WIDTH/8):1,
    parameter ALMOST_FULL_OFFSET = 13'h0005, //threshold for almost full signal
    parameter ALMOST_EMPTY_OFFSET = 13'h0005) //threshold for almost empty signal
(
    input wire in_clk,
    input wire in_rst,
    input wire [DATA_WIDTH-1:0] in_tdata,
    input wire [KEEP_WIDTH-1:0] in_tkeep,
    input wire [USER_WIDTH-1:0] in_tuser,
    input wire in_tlast,
    input wire in_tvalid,
    output wire in_tready,

    input wire out_clk,
    input wire out_rst,
    output wire [DATA_WIDTH-1:0] out_tdata,
    output wire [KEEP_WIDTH-1:0] out_tkeep,
    output wire [USER_WIDTH-1:0] out_tuser,
    output wire out_tlast,
    output wire out_tvalid,
    input wire out_tready,

    output wire [DATA_COUNT_WIDTH-1:0] wr_count,
    output wire [DATA_COUNT_WIDTH-1:0] rd_count,

    output wire almost_empty,
    output wire almost_full
);
    localparam FULL_WIDTH = DATA_WIDTH + KEEP_WIDTH;

    wire [DATA_WIDTH-1:0] wfifo_tdata;
    wire [KEEP_WIDTH-1:0] wfifo_tkeep, wfifo_tkeep_i;
    wire [USER_WIDTH-1:0] wfifo_tuser;
    wire wfifo_tlast;
    wire wfifo_tvalid;
    wire wfifo_tready;

    reg in_rd_burst = 1'b1;
    reg in_wr_burst = 1'b1;

    generate
    //block ram fifo: insert small LUT fifos for timing
    //or with tuser to remove fifo deadlocks for tuser interleave
    if (FIFO_MEMORY_TYPE == "block" || ENABLE_TUSER) begin
    stream_fifo_srl32 #(.DATA_WIDTH(USER_WIDTH+FULL_WIDTH)) in_fifo
    (
        .clk(in_clk),
        .rst(in_rst),
        .in_tdata({in_tuser, in_tkeep, in_tdata}),
        .in_tlast(in_tlast),
        .in_tvalid(in_tvalid),
        .in_tready(in_tready),
        .out_tdata({wfifo_tuser, wfifo_tkeep, wfifo_tdata}),
        .out_tlast(wfifo_tlast),
        .out_tvalid(wfifo_tvalid),
        .out_tready(wfifo_tready)
    );
    end
    //otherwise connect directly
    else begin
        assign wfifo_tdata = in_tdata;
        assign wfifo_tkeep = in_tkeep;
        assign wfifo_tuser = in_tuser;
        assign wfifo_tlast = in_tlast;
        assign wfifo_tvalid = in_tvalid;
        assign in_tready = wfifo_tready;
    end
    endgenerate

    wire [DATA_WIDTH-1:0] rfifo_tdata;
    wire [KEEP_WIDTH-1:0] rfifo_tkeep, rfifo_tkeep_i;
    wire rfifo_tlast;
    wire rfifo_tvalid;
    wire rfifo_tready;

    wire empty;
    wire full;
    wire wr_rst_busy;
    wire rd_rst_busy;
    wire read = rfifo_tready && !empty && !rd_rst_busy;
    wire write = wfifo_tvalid && !full && !wr_rst_busy;
    assign wfifo_tready = !full && !wr_rst_busy && in_wr_burst;
    assign rfifo_tvalid = !empty && !rd_rst_busy && in_rd_burst;

    wire [KEEP_WIDTH-1:0] tkeep_zeros = {KEEP_WIDTH{1'b0}};

    //encode last into tkeep: 0s on all but last transfer cycles
    //and tkeep cannot be all zero on last, will be inverted
    generate
    if (ENABLE_TKEEP) assign wfifo_tkeep_i = wfifo_tlast?((|(wfifo_tkeep))?wfifo_tkeep:(~tkeep_zeros)):tkeep_zeros;
    else              assign wfifo_tkeep_i = wfifo_tlast?~tkeep_zeros:tkeep_zeros;
    endgenerate

    //store the first write as tuser
    wire [DATA_WIDTH-1:0] wfifo_tdata_i = in_wr_burst?wfifo_tdata:wfifo_tuser;
    generate
    if (ENABLE_TUSER)
        always @(posedge in_clk)
            if (in_rst) in_wr_burst <= 1'b0;
            else if (write) in_wr_burst <= in_wr_burst?!wfifo_tlast:1'b1;
    endgenerate

    //decode output tkeep and last
    //last is signaled by a valid tkeep output
    //output tkeep is all 1s until last transfer
    assign rfifo_tlast = |(rfifo_tkeep_i);
    assign rfifo_tkeep = rfifo_tlast?rfifo_tkeep_i:~tkeep_zeros;

    localparam DATA_COUNT_WIDTH_INT = $clog2(FIFO_DEPTH)+1;
    generate
    if (DATA_COUNT_WIDTH > DATA_COUNT_WIDTH_INT) begin
        assign wr_count[DATA_COUNT_WIDTH-1:DATA_COUNT_WIDTH_INT] = 0;
        assign rd_count[DATA_COUNT_WIDTH-1:DATA_COUNT_WIDTH_INT] = 0;
    end
    endgenerate

    xpm_fifo_async #(
        .FIFO_MEMORY_TYPE(FIFO_MEMORY_TYPE),
        .FIFO_WRITE_DEPTH(FIFO_DEPTH),
        .RELATED_CLOCKS(0),
        .WRITE_DATA_WIDTH(FULL_WIDTH),
        .WR_DATA_COUNT_WIDTH(DATA_COUNT_WIDTH_INT),
        .READ_MODE("fwft"),
        .FIFO_READ_LATENCY(0),
        .READ_DATA_WIDTH(FULL_WIDTH),
        .RD_DATA_COUNT_WIDTH(DATA_COUNT_WIDTH_INT),
        .PROG_FULL_THRESH(FIFO_DEPTH-ALMOST_FULL_OFFSET),
        .PROG_EMPTY_THRESH(ALMOST_EMPTY_OFFSET))
    fifo(
        .wr_clk(in_clk),
        .wr_en(write),
        .din({wfifo_tkeep_i, wfifo_tdata_i}),
        .full(full),
        .overflow(),
        .wr_rst_busy(wr_rst_busy),
        .sleep(1'b0),
        .rst(in_rst),
        .rd_clk(out_clk),
        .rd_en(read),
        .dout({rfifo_tkeep_i, rfifo_tdata}),
        .empty(empty),
        .underflow(),
        .rd_rst_busy(rd_rst_busy),
        .prog_full(almost_full),
        .wr_data_count(wr_count[DATA_COUNT_WIDTH-1:0]),
        .wr_ack(),
        .prog_empty(almost_empty),
        .rd_data_count(rd_count[DATA_COUNT_WIDTH-1:0]),
        .injectsbiterr(1'b0),
        .injectdbiterr(1'b0),
        .sbiterr(),
        .almost_empty(),
        .data_valid(),
        .almost_full(),
        .dbiterr());

    //store the first read as tuser
    reg [USER_WIDTH-1:0] rfifo_tuser;
    generate
    if (ENABLE_TUSER)
        always @(posedge out_clk)
            if (out_rst) in_rd_burst <= 1'b0;
            else if (read) begin
                in_rd_burst <= in_rd_burst?!rfifo_tlast:1'b1;
                rfifo_tuser <= in_rd_burst?{USER_WIDTH{1'b0}}:rfifo_tdata;
            end
    endgenerate

    generate
    //block ram fifo: insert small LUT fifos for timing
    //or with tuser to remove fifo deadlocks for tuser interleave
    if (FIFO_MEMORY_TYPE == "block" || ENABLE_TUSER) begin
    stream_fifo_srl32 #(.DATA_WIDTH(USER_WIDTH+FULL_WIDTH)) out_fifo
    (
        .clk(out_clk),
        .rst(out_rst),
        .in_tdata({rfifo_tuser, rfifo_tkeep, rfifo_tdata}),
        .in_tlast(rfifo_tlast),
        .in_tvalid(rfifo_tvalid),
        .in_tready(rfifo_tready),
        .out_tdata({out_tuser, out_tkeep, out_tdata}),
        .out_tlast(out_tlast),
        .out_tvalid(out_tvalid),
        .out_tready(out_tready)
    );
    end
    //otherwise connect directly
    else begin
        assign out_tdata = rfifo_tdata;
        assign out_tkeep = rfifo_tkeep;
        assign out_tuser = rfifo_tuser;
        assign out_tlast = rfifo_tlast;
        assign out_tvalid = rfifo_tvalid;
        assign rfifo_tready = out_tready;
    end
    endgenerate

endmodule //cdc_stream_fifo

`default_nettype wire
