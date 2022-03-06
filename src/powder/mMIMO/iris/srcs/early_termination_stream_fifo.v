/////////////////////////////////////////////////////////////////////
// Stream fifo forwarder with early temination capability
//
// The stream fifo has one element of history
// so we can terminate the output with tlast when forced
// even if the transfer held in history was not a tlast.
// Use this to safely terminate a packet without
// writing out an extra cycle of junk.
/////////////////////////////////////////////////////////////////////

`default_nettype none

module early_termination_stream_fifo #(
    parameter DATA_WIDTH = 64)
(
    input wire clk,
    input wire rst,

    input wire terminate,

    input wire [DATA_WIDTH-1 : 0] in_tdata,
    input wire in_tlast,
    output wire in_tready,
    input wire in_tvalid,

    output wire [DATA_WIDTH-1 : 0] out_tdata,
    output wire out_tlast,
    input wire out_tready,
    output wire out_tvalid
);

    reg [DATA_WIDTH-1:0] hist_data;
    reg hist_valid;
    reg hist_last;

    wire out_tready0, out_tvalid0, out_tlast0;

    wire wr_xfer = (in_tready && in_tvalid);
    wire rd_xfer = (out_tready0 && out_tvalid0);

    always @(posedge clk) begin
        if (rst) begin
            hist_data <= 0;
            hist_valid <= 1'b0;
            hist_last <= 1'b0;
        end

        //writing only (or writing and reading)
        else if (wr_xfer) begin
            hist_data <= in_tdata;
            hist_valid <= in_tvalid;
            hist_last <= in_tlast || terminate;
        end

        //reading only (clears history)
        else if (rd_xfer) begin
            hist_data <= 0;
            hist_valid <= 1'b0;
            hist_last <= 1'b0;
        end

        //terminate cycle, but we didnt read/write
        //re-write the history to be a tlast cycle
        else if (terminate && hist_valid) begin
            hist_last <= 1'b1;
        end
    end

    wire [DATA_WIDTH-1:0] out_tdata0 = hist_data;

    assign out_tlast0 = hist_last || (terminate && !wr_xfer);

    //output is valid when there is history and there is a new input or flush condition
    assign out_tvalid0 = hist_valid && (in_tvalid || out_tlast0);

    //input is ready when there is no history, or output is ready
    assign in_tready = hist_valid? out_tready0 : 1'b1;

    stream_fifo_srl32 #(.DATA_WIDTH(DATA_WIDTH))
    u_out_fifo(
        .clk(clk),
        .rst(rst),
        .in_tdata(out_tdata0),
        .in_tlast(out_tlast0),
        .in_tvalid(out_tvalid0),
        .in_tready(out_tready0),
        .out_tdata(out_tdata),
        .out_tlast(out_tlast),
        .out_tvalid(out_tvalid),
        .out_tready(out_tready));

endmodule //early_termination_stream_fifo

`default_nettype wire
