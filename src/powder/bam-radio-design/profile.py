#!/usr/bin/python


# Library imports
import geni.portal as portal
import geni.rspec.pg as rspec
import geni.rspec.emulab.pnext as pn
import geni.rspec.emulab.spectrum as spectrum
import geni.rspec.igext as ig


# Global Variables
x310_node_disk_image = \
        "urn:publicid:IDN+emulab.net+image+emulab-ops//UBUNTU18-64-STD"
setup_command = "/local/repository/startup.sh"
installs = ["gnuradio"]

# Top-level request object.
request = portal.context.makeRequestRSpec()

# Helper function that allocates a PC + X310 radio pair, with Ethernet
# link between them.
def x310_node_pair(idx, x310_radio_name, node_type, installs):
    radio_link = request.Link("radio-link-%d" % idx)

    node = request.RawPC("%s-comp" % x310_radio_name)
    node.hardware_type = node_type
    node.disk_image = x310_node_disk_image

    service_command = " ".join([setup_command] + installs)
    node.addService(rspec.Execute(shell="bash", command=service_command))

    node_radio_if = node.addInterface("usrp_if")
    node_radio_if.addAddress(rspec.IPv4Address("192.168.40.1",
                                               "255.255.255.0"))
    radio_link.addInterface(node_radio_if)

    radio = request.RawPC("%s-x310" % x310_radio_name)
    radio.component_id = x310_radio_name
    radio_link.addNode(radio)

# Node type parameter for PCs to be paired with X310 radios.
# Restricted to those that are known to work well with them.
portal.context.defineParameter(
    "nodetype",
    "Compute node type",
    portal.ParameterType.STRING, "d740",
    ["d740","d430"],
    "Type of compute node to be paired with the X310 Radios",
)

# List of Cellular rooftop X310 radios.
rooftop_names = [
    ("cellsdr1-bes",
     "Behavioral"),
    ("cellsdr1-ustar",
     "USTAR"),
]

    
# Multi-value list of x310+PC pairs to add to experiment.
portal.context.defineStructParameter(
    "radios", "X310 Cellular Radios",
    multiValue=False,
    members=[
        portal.Parameter(
            "radio_name1",
            "Rooftop base-station X310 #1",
            portal.ParameterType.STRING,
            rooftop_names[0],
            rooftop_names),
        portal.Parameter(
            "radio_name2",
            "Rooftop base-station X310 #2",
            portal.ParameterType.STRING,
            rooftop_names[1],
            rooftop_names)
    ])

# Bind and verify parameters
params = portal.context.bindParameters()

portal.context.verifyParameters()

# Request frequency range(s)
request.requestSpectrum(2530.0, 2540.0, 0)
request.requestSpectrum(2510.0, 2520.0, 0)

# Request PC + X310 resource pairs.
x310_node_pair(1, params.radios.radio_name1, params.nodetype, installs)
x310_node_pair(2, params.radios.radio_name2, params.nodetype, installs)

# Emit!
portal.context.printRequestRSpec()