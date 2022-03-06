#!/usr/bin/python

"""This profile allows the allocation of resources for over-the-air
operation on the POWDER platform. Specifically, the profile has
options to request the allocation of SDR radios in rooftop 
base-stations and fixed-endpoints (i.e., nodes deployed at
human height).

Map of deployment is here:
https://www.powderwireless.net/map

The base-station SDRs are X310s and connected to an antenna
covering the cellular band (1695 - 2690 MHz), i.e., cellsdr,
or to an antenna covering the CBRS band (3400 - 3800 MHz), i.e.,
cbrssdr. Each X310 is paired with a compute node (by default
a Dell d740).

The fixed-endpoint SDRs are B210s each of which is paired with 
an Intel NUC small form factor compute node. Both B210s are connected
to broadband antennas: nuc1 is connected in an RX only configuration,
while nuc2 is connected in a TX/RX configuration.

The profile uses a disk image with srsLTE software, as well as
GNU Radio and the UHD software tools, pre-installed.

Resources needed to realize a basic srsLTE setup consisting of a UE, an eNodeB and an EPC core network:

  * Frequency ranges (uplink and downlink) for LTE FDD operation. 
  * A "nuc2" fixed-end point compute/SDR pair. (This will run the UE side.)
  * A "cellsdr" base station SDR. (This will be the radio side of the eNodeB.)
  * A "d740" compute node. (This will run both the eNodeB software and the EPC software.)
  
**Specific resources that can be used (and that need to be reserved before instantiating the profile):** 

  * Hardware (at least one set of resources are needed):
   * Humanities, nuc2; Emulab, cellsdr1-browning; Emulab, d740
   * Bookstore, nuc2; Emulab, cellsdr1-bes; Emulab, d740
   * Moran, nuc2; Emulab, cellsdr1-ustar; Emulab, d740 
  * Frequencies:
   * Uplink frequency: 2530 MHz to 2540 MHz
   * Downlink frequency: 2640 MHz to 2650 MHz

The instuctions below assume the first hardware configuration.

Instructions:

The instructions below assume the following hardware set was selected when the profile was instantiated:

 * Bookstore, nuc2; Emulab, cellsdr1-browning; Emulab, d740

#### To run the srsLTE software

**To run the EPC**

Open a terminal on the `cellsdr1-browning-comp` node in your experiment. (Go to the "List View"
in your experiment. If you have ssh keys and an ssh client working in your
setup you should be able to click on the black "ssh -p ..." command to get a
terminal. If ssh is not working in your setup, you can open a browser shell
by clicking on the Actions icon corresponding to the node and selecting Shell
from the dropdown menu.)

Start up the EPC:

    sudo srsepc
    
**To run the eNodeB**

Open another terminal on the `cellsdr1-browning-comp` node in your experiment.

Start up the eNodeB:

    sudo srsenb

**To run the UE**

Open a terminal on the `b210-bookstore-nuc2` node in your experiment.

Start up the UE:

    sudo srsue

**Verify functionality**

Open another terminal on the `b210-bookstore-nuc2` node in your experiment.

Verify that the virtual network interface tun_srsue" has been created:

    ifconfig tun_srsue

Run ping to the SGi IP address via your RF link:
    
    ping 172.16.0.1

Killing/restarting the UE process will result in connectivity being interrupted/restored.

If you are using an ssh client with X11 set up, you can run the UE with the GUI
enabled to see a real time view of the signals received by the UE:

    sudo srsue --gui.enable 1

Note: If srsenb fails with an error indicating "No compatible RF-frontend
found", you'll need to flash the appropriate firmware to the X310 and
power-cycle it using the portal UI. Run `uhd_usrp_probe` in a shell on the
associated compute node to get instructions for downloading and flashing the
firmware. Use the Action buttons in the List View tab of the UI to power cycle
the appropriate X310. If srsue fails with a similar error, try power-cycling the
associated NUC.

"""

import geni.portal as portal
import geni.rspec.pg as rspec
import geni.rspec.emulab.pnext as pn
import geni.rspec.igext as ig
import geni.rspec.emulab.spectrum as spectrum


class GLOBALS:
    SRSLTE_IMG = "urn:publicid:IDN+emulab.net+image+PowderTeam:U18LL-SRSLTE:3"
    SRSLTE_SRC_DS = "urn:publicid:IDN+emulab.net:powderteam+imdataset+srslte-src-v19"
    DLHIFREQ = 2650.0
    DLLOFREQ = 2640.0
    ULHIFREQ = 2540.0
    ULLOFREQ = 2530.0


def x310_node_pair(idx, x310_radio):
    radio_link = request.Link("radio-link-%d"%(idx))
    radio_link.bandwidth = 10*1000*1000

    node = request.RawPC("%s-comp"%(x310_radio.radio_name))
    node.hardware_type = params.x310_pair_nodetype
    node.disk_image = GLOBALS.SRSLTE_IMG
    node.component_manager_id = "urn:publicid:IDN+emulab.net+authority+cm"
    node.addService(rspec.Execute(shell="bash", command="/local/repository/bin/add-nat-and-ip-forwarding.sh"))
    node.addService(rspec.Execute(shell="bash", command="/local/repository/bin/update-config-files.sh"))
    node.addService(rspec.Execute(shell="bash", command="/local/repository/bin/tune-cpu.sh"))
    node.addService(rspec.Execute(shell="bash", command="/local/repository/bin/tune-sdr-iface.sh"))

    if params.include_srslte_src:
        bs = node.Blockstore("bs-comp-%s"%idx, "/opt/srslte")
        bs.dataset = GLOBALS.SRSLTE_SRC_DS

    node_radio_if = node.addInterface("usrp_if")
    node_radio_if.addAddress(rspec.IPv4Address("192.168.40.1",
                                               "255.255.255.0"))
    radio_link.addInterface(node_radio_if)

    radio = request.RawPC("%s-x310"%(x310_radio.radio_name))
    radio.component_id = x310_radio.radio_name
    radio_link.addNode(radio)


def b210_nuc_pair(idx, b210_node):
    b210_nuc_pair_node = request.RawPC("b210-%s-%s"%(b210_node.aggregate_id,"nuc2"))
    agg_full_name = "urn:publicid:IDN+%s.powderwireless.net+authority+cm"%(b210_node.aggregate_id)
    b210_nuc_pair_node.component_manager_id = agg_full_name
    b210_nuc_pair_node.component_id = "nuc2"
    b210_nuc_pair_node.disk_image = GLOBALS.SRSLTE_IMG
    b210_nuc_pair_node.addService(rspec.Execute(shell="bash", command="/local/repository/bin/update-config-files.sh"))
    b210_nuc_pair_node.addService(rspec.Execute(shell="bash", command="/local/repository/bin/tune-cpu.sh"))

    if params.include_srslte_src:
        bs = b210_nuc_pair_node.Blockstore("bs-nuc-%s"%idx, "/opt/srslte")
        bs.dataset = GLOBALS.SRSLTE_SRC_DS


portal.context.defineParameter("include_srslte_src",
                               "Include srsLTE source code.",
                               portal.ParameterType.BOOLEAN,
                               False)

node_type = [
    ("d740",
     "Emulab, d740"),
    ("d430",
     "Emulab, d430")
]

portal.context.defineParameter("x310_pair_nodetype",
                               "Type of compute node paired with the X310 Radios",
                               portal.ParameterType.STRING,
                               node_type[0],
                               node_type)

rooftop_names = [
    ("cellsdr1-bes",
     "Emulab, cellsdr1-bes (Behavioral)"),
    ("cellsdr1-browning",
     "Emulab, cellsdr1-browning (Browning)"),
    ("cellsdr1-dentistry",
     "Emulab, cellsdr1-dentistry (Dentistry)"),
    ("cellsdr1-fm",
     "Emulab, cellsdrsdr1-fm (Friendship Manor)"),
    ("cellsdr1-honors",
     "Emulab, cellsdrs1-honors (Honors)"),
    ("cellsdr1-meb",
     "Emulab, cellsdr1-meb (MEB)"),
    ("cellsdr1-smt",
     "Emulab, cellsdrsdr1-smt (SMT)"),
    ("cellsdr1-ustar",
     "Emulab, cellsdr1-ustar (USTAR)")
]

portal.context.defineStructParameter("x310_radios", "X310 Radios", [],
                                     multiValue=True,
                                     itemDefaultValue=
                                     {},
                                     min=0, max=None,
                                     members=[
                                        portal.Parameter(
                                             "radio_name",
                                             "Rooftop base-station X310",
                                             portal.ParameterType.STRING,
                                             rooftop_names[0],
                                             rooftop_names)
                                     ])

fixed_endpoint_aggregates = [
    ("web",
     "WEB, nuc2"),
    ("ebc",
     "EBC, nuc2"),
    ("bookstore",
     "Bookstore, nuc2"),
    ("humanities",
     "Humanities, nuc2"),
    ("law73",
     "Law 73, nuc2"),
    ("madsen",
     "Madsen, nuc2"),
    ("sagepoint",
     "Sage Point, nuc2"),
    ("moran",
     "Moran, nuc2"),
]

portal.context.defineStructParameter("b210_nodes", "B210 Radios", [],
                                     multiValue=True,
                                     min=0, max=None,
                                     members=[
                                         portal.Parameter(
                                             "aggregate_id",
                                             "Fixed Endpoint B210",
                                             portal.ParameterType.STRING,
                                             fixed_endpoint_aggregates[0],
                                             fixed_endpoint_aggregates)
                                     ],
                                    )


params = portal.context.bindParameters()
request = portal.context.makeRequestRSpec()
request.requestSpectrum(GLOBALS.ULLOFREQ, GLOBALS.ULHIFREQ, 0)
request.requestSpectrum(GLOBALS.DLLOFREQ, GLOBALS.DLHIFREQ, 0)

for i, x310_radio in enumerate(params.x310_radios):
    x310_node_pair(i, x310_radio)

for i, b210_node in enumerate(params.b210_nodes):
    b210_nuc_pair(i, b210_node)

portal.context.printRequestRSpec()
