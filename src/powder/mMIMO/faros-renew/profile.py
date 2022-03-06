"""This profile instantiates a d840 machine connected to a Skylark FAROS massive MIMO system comprised of a FAROS hub, a Faros massive MIMO Base Station, and a set of Iris UEs (clients). 
The PC boots with Ubuntu 18.04 and includes a MATLAB installation that could be used to run experiments on FAROS with RENEWLab demos. For more information on RENEWLab, see [RENEW documentation page](https://wiki.renew-wireless.org/)

Instructions:
The FAROS hub and PC are connected via a private 10Gbps link. 
All Iris radios and the FAROS hub should come up with address between 192.168.1.101 and 192.168.1.200. 
These addresses are reachable by first logging in to the PC. 
For more information on how to start an experiment with massive MIMO equipment on POWDER, see [this page](https://wiki.renew-wireless.org/en/quickstartguide/userenewonpowder). 

To get started with FAROS massive MIMO hardware, see [RENEW Hardware Documentation](https://wiki.renew-wireless.org/en/architectures/hardwarearchitecture).

For questions about access to the required PC type and massive MIMO radio devices, please contact support@powderwireless.net

- Once your experiment is ready, from your terminal, ssh to pc1 with X11 forwarding:
 
	`ssh -X USERNAME@pc19-meb.emulab.net`

- The RENEWLab code repository will be automatically cloned in the /scratch/ directory: 

	`cd /scratch/RENEWLab`

- Follow the instructions in our [wiki](https://wiki.renew-wireless.org/en/quickstartguide/userenewonpowder) to run a demo script.  

"""

import geni.portal as portal
import geni.urn as urn
import geni.rspec.pg as pg
import geni.rspec.emulab as elab
import geni.rspec.emulab.spectrum as spectrum

# Resource strings
PCIMG = "urn:publicid:IDN+emulab.net+image+emulab-ops:UBUNTU18-64-STD"
MATLAB_DS_URN = "urn:publicid:IDN+emulab.net:powdersandbox+imdataset+matlab2021ra-etc" # "urn:publicid:IDN+emulab.net:powderprofiles+ltdataset+matlab-extra"
MATLAB_MP = "/usr/local/MATLAB"
STARTUP_SCRIPT = "/local/repository/faros_start.sh"
PCHWTYPE = "d840"
FAROSHWTYPE = "faros_sfp"
IRISHWTYPE = "iris030"
DEF_BS_SIZE = 0
DEF_BS_MOUNT_POINT = "/opt/data"
DEF_REMDS_MP = "/opt/data"

REMDS_TYPES = [("readwrite", "Read-Write (persistent)"),
               ("readonly", "Read Only"),
               ("rwclone", "Read-Write Clone (not persistent)")]

MMIMO_ARRAYS = ["", ("mmimo-ac", "Anechoic chamber array"),
                ("mmimo1", "Honors rooftop array")]

LAB_CLIENTS = ["", ("irisclients-ac", "Anechoic chamber clients")]

#
# Profile parameters.
#
pc = portal.Context()

# Array to allocate
pc.defineParameter("mmimoid", "Name of  Massive MIMO array to allocate.",
                   portal.ParameterType.STRING, MMIMO_ARRAYS[0], MMIMO_ARRAYS,
                   longDescription="Leave blank to omit mMIMO array.")

# Allocate client radio?
pc.defineParameter("labclient", "Allocate lab Iris client radio.",
                   portal.ParameterType.STRING, LAB_CLIENTS[0], LAB_CLIENTS,
                   longDescription="Leave blank to omit lab client.")

# Frequency/spectrum parameters
pc.defineStructParameter(
    "freq_ranges", "Range", [],
    multiValue=True,
    min=1,
    multiValueTitle="Frequency ranges for over-the-air operation.",
    members=[
        portal.Parameter(
            "freq_min",
            "Frequency Min",
            portal.ParameterType.BANDWIDTH,
            3550.0,
            longDescription="Values are rounded to the nearest kilohertz."
        ),
        portal.Parameter(
            "freq_max",
            "Frequency Max",
            portal.ParameterType.BANDWIDTH,
            3560.0,
            longDescription="Values are rounded to the nearest kilohertz."
        ),
    ])

pc.defineParameter("fixedid", "Fixed PC Node_id (Optional)",
                   portal.ParameterType.STRING, "", advanced=True,
                   longDescription="Fix 'pc1' to this specific node.  Leave blank to allow for any available node of the correct type.")

pc.defineParameter("matlabds", "Attach the Matlab dataset to the compute host.",
                   portal.ParameterType.BOOLEAN, True, advanced=True)

pc.defineParameter("hubints", "Number of interfaces to attach on hub (def: 2)",
                   portal.ParameterType.INTEGER, 2, advanced=True,
                   longDescription="This can be a number between 1 and 4.")

pc.defineParameter("remds", "Remote Dataset (Optional)",
                   portal.ParameterType.STRING, "", advanced=True,
                   longDescription="Insert URN of a remote dataset to mount. Leave blank for no dataset.")

pc.defineParameter("remmp", "Remote Dataset Mount Point (Optional)",
                   portal.ParameterType.STRING, DEF_REMDS_MP, advanced=True,
                   longDescription="Mount point for optional remote dataset.  Ignored if the 'Remote Dataset' parameter is not set.")

pc.defineParameter("remtype", "Remote Dataset Mount Type (Optional)",
                   portal.ParameterType.STRING, REMDS_TYPES[0], REMDS_TYPES,
                   advanced=True, longDescription="Type of mount for remote dataset.  Ignored if the 'Remote Dataset' parameter is not set.")

# Bind and verify parameters.
params = pc.bindParameters()

for i, frange in enumerate(params.freq_ranges):
    if frange.freq_max - frange.freq_min < 1:
        perr = portal.ParameterError("Minimum and maximum frequencies must be separated by at least 1 MHz", ["freq_ranges[%d].freq_min" % i, "freq_ranges[%d].freq_max" % i])
        portal.context.reportError(perr)

if params.hubints < 1 or params.hubints > 4:
    perr = portal.ParameterError("Number of interfaces on hub to connect must be between 1 and 4 (inclusive).")
    portal.context.reportError(perr)

pc.verifyParameters()

# Create a Request object to start building the RSpec.
request = pc.makeRequestRSpec()

# Mount a remote dataset
def connect_DS(node, urn, mp, dsname = "", dstype = "rwclone"):
    if not dsname:
        dsname = "ds-%s" % node.name
    bs = request.Blockstore(dsname, mp)
    if dstype == "rwclone":
        bs.rwclone = True
    elif dstype == "readonly":
        bs.readonly = True
        
    # Set dataset URN
    bs.dataset = urn

    # Create link from node to OAI dataset rw clone
    bslink = request.Link("link_%s" % dsname, members=(node, bs.interface))
    bslink.vlan_tagging = True
    bslink.best_effort = True

# Request a PC
pc1 = request.RawPC("pc1")
if params.fixedid:
    pc1.component_id=params.fixedid
else:
    pc1.hardware_type = PCHWTYPE
pc1.disk_image = PCIMG
if params.matlabds:
    # connect_DS(pc1, MATLAB_DS_URN, MATLAB_MP, "matlab1")
    mlbs = pc1.Blockstore( "matlab1", MATLAB_MP )
    mlbs.dataset = MATLAB_DS_URN
    mlbs.placement = "nonsysvol"

pc1.addService(pg.Execute(shell="sh", command=STARTUP_SCRIPT))
if1pc1 = pc1.addInterface("if1pc1", pg.IPv4Address("192.168.1.1", "255.255.255.0"))
#if1pc1.bandwidth = 40 * 1000 * 1000 # 40 Gbps
bs2 = pc1.Blockstore("scratch","/scratch")
bs2.size = "100GB"
bs2.placement = "nonsysvol"
if params.remds:
    connect_DS(pc1, params.remds, params.remmp, dstype=params.remtype)

# LAN connecting up everything (if needed).  Members are added below.
lan1 = None
if params.mmimoid or params.labclient:
    lan1 = request.LAN("lan1")
    lan1.vlan_tagging = False
    lan1.setNoBandwidthShaping()
    lan1.addInterface(if1pc1)

# Request a Faros BS.
if params.mmimoid:
    mm1 = request.RawPC("mm1")
    mm1.component_id = params.mmimoid
    #mm1.hardware_type = FAROSHWTYPE
    for i in range(params.hubints):
        mmif = mm1.addInterface()
        lan1.addInterface(mmif)

# Lab client to allocate (if any).
if params.labclient:
    labir = request.RawPC("labir1")
    #labir.hardware_type = IRISHWTYPE
    labir.component_id = params.labclient
    labif = labir.addInterface()
    lan1.addInterface(labif)

# Add frequency request(s)
for frange in params.freq_ranges:
    request.requestSpectrum(frange.freq_min, frange.freq_max, 100)

# Print the RSpec to the enclosing page.
pc.printRequestRSpec()
