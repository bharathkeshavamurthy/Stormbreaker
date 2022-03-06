//----------------------------------------------------------------------
//-- handshake_fifo for clock domain crossing with axi stream
//--
//-- Cross clock domains using simple handshake logic with backpressure.
//-- Use this block when crossing domains for low speed control and status.
//----------------------------------------------------------------------

`default_nettype none

module handshake_fifo # (
    parameter DATA_WIDTH = 32) //threshold for almost empty signal
(
    input wire in_clk,
    input wire in_rst,
    input wire [DATA_WIDTH-1:0] in_tdata,
    input wire in_tlast,
    input wire in_tvalid,
    output wire in_tready,

    input wire out_clk,
    input wire out_rst,
    output wire [DATA_WIDTH-1:0] out_tdata,
    output wire out_tlast,
    output wire out_tvalid,
    input wire out_tready
);

    reg src_send;
    wire src_rcv;
    wire dest_req;
    reg dest_ack;

    reg [DATA_WIDTH:0] src_in;
    wire [DATA_WIDTH:0] dest_out;

    //----------- input handshake logic between axi and handshaker ------------//
    always @(posedge in_clk) begin
        if (in_rst) src_send <= 1'b0;

        //got a new value, register it, disable the input bus, and raise send
        else if (in_tvalid && in_tready) src_send <= 1'b1;

        //src receive was asserted, deassert send
        else if (src_send && src_rcv) src_send <= 1'b0;

        //hold the input value on a valid transfer
        if (in_rst) src_in <= 0;
        else if (in_tvalid && in_tready) src_in <= {in_tlast, in_tdata};
    end

    assign in_tready = !src_send && !src_rcv;

    //----------- output handshake logic between handshaker and axi ------------//
    always @(posedge out_clk) begin
        if (out_rst) dest_ack <= 1'b0;

        //the transfer completed, raise the ack
        else if (out_tvalid && out_tready) dest_ack <= 1'b1;

        //the dst request lowered, now deassert ack
        else if (!dest_req && dest_ack) dest_ack <= 1'b0;
    end

    assign out_tvalid = dest_req && !dest_ack;
    assign {out_tlast, out_tdata} = dest_out;

    xpm_cdc_handshake #(
        .DEST_EXT_HSK   ( 1           ),
        .DEST_SYNC_FF   ( 2           ),
        .SRC_SYNC_FF    ( 2           ),
        .WIDTH          ( DATA_WIDTH + 1 )
    ) handshake (
        .src_clk  ( in_clk         ),
        .src_in   ( src_in         ),
        .src_send ( src_send       ),
        .src_rcv  ( src_rcv        ),

        .dest_clk ( out_clk        ),
        .dest_out ( dest_out       ),
        .dest_req ( dest_req       ),
        .dest_ack ( dest_ack       )
    );

endmodule //handshake_fifo

`default_nettype wire
