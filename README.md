# ntuple-tools

The python scripts in this repository should help you get started analysing the [HGCAL L1 TP ntuples](https://github.com/PFCal-dev/cmssw/tree/hgc-tpg-devel-CMSSW_10_3_0_pre4/L1Trigger/L1THGCal/plugins/ntuples)

## Pre-requisites

### 1. setup virtualenvwrapper

Setup a `virtualenv` using `virtualenvwrapper`.

Follow the `virtualenvwrapper` [installation instructions](https://virtualenvwrapper.readthedocs.io/en/latest/install.html) to install it in the `~/.local/` directory (using `$ pip install --user virtualenvwrapper`). This needs to be done only once for your account and can be done with whatever `python` version is in use in the system.

For starting using virtualenvwrapper

`source setVirtualEnvWrapper.sh`

### 2. lxplus setup

This step is `lxplus` specific, givin access to a more recent `python` and `root` version.
Edit/skip it accordingly for your specific system.

`source setup_lxplus.sh`

### 3. create a virtualenv for the project

The first time you will have to create the actual instance of the `virtualenv`:

`mkvirtualenv -p \`which python3.8\` <venvname>`

The requirements for the virtualenv setup are in are in the file:

[requirements.txt](requirements.txt)

You can use the file directly using:

`pip install -r requirements.txt`

If you want to use python 3 (recommended)

`mkvirtualenv -p <path_python3.8> <venvname>`

`pip install -r requirements_py3.8.txt`

After this initial (once in a time) setup is done you can just activate the virtualenv calling:

`workon  <venvname>`

(`lsvirtualenv` is your friend in case you forgot the name).


## HGCAL L1 TPG analysis

`python analyzeHgcalL1Tntuple.py -f cfg/default.yaml -i cfg/datasets/ntp_v68C.yaml -c decoded_tk -s doubleele_flat1to100_PU200 -n 3000 -d 2`

see:

`python analyzeHgcalL1Tntuple.py --help`

for the details.

### Configuration file
The configuration is handled by 2 yaml files. 
One specifying    
   - output directories
   - versioning of the plots
   - collections of samples, i.e. group of samples to be processed homogeneously: for each collection the list of plotters (see below) to be run is provided.

The other prividing
   - details of the input samples (location of the ntuple files)

Example of configuration file can be found in:
 - [cfg/default.yaml](cfg/default.yaml)
 - [cfg/datasets/ntp_v66.yaml](cfg/datasets/ntp_v66.yaml)


### Reading ntuple branches or creating derived ones
The list of branches to be read and converted in pandas `DataFrame` format is specified in the module

[collections](python/collections.py)

Instantiating an object of class `DFCollection`. What is actually read event by event depends anyhow on which plotters are actually instantiated (collections are read on-demand).

### Selecting subsets of object collections
Selections are defined as strings in the module:

[selections](python/selections.py)

Different collections are defined for different objects and/or different purposes. The selections have a `name` whcih is used for the histogram naming (see below). Selections are used by the plotters.


### Adding a new plotter
The actual functionality of accessing the objects, filtering them according to the `selections` and filling `histograms` is provided by the plotter classes defined in the module:

[plotters](python/plotters.py)

Basic plotters are already available, most likely you just need to instantiate one of them (or a collection of them) using the `DFCollection` instance you are interested in.
Which collection is run for which sample is steered by the configuration file.

The plotters access one or more collections, select them in several different ways, book and fill the histograms (see below).

### Adding a new histogram
Histograms are handled in the module:

[l1THistos](python/l1THistos.py)

There are different classes of histograms depending on the input object and on the purpose.
To add a new histogram to an existing class it is enough to add it in the corresponding constructor and in the `fill` module. The writing of the histos to files is handled transparently.

The histogram naming follows the convention:
`<ObjectName>_<SelectionName>_<GenSelectionName>_<HistoName>`

This is assumed in all the `plotters` and in the code to actually draw the histograms.


## Submitting to the batch system

Note that the script `analyzeHgcalL1Tntuple.py` can be used to submit the jobs to the HTCondor batch system invoking the `-b` option. A dag configuration is created and you can actually submit it following the script output.

### Note about hadd job.
For each sample injected in the batch system a DAG is created. The DAG will submitt an `hadd` command once all the jobs will succeed.
However, if you don't want to wait (or you don't care) you can submit also a condor job that will run hadd periodically thus reducing dramatically the latency.
For example:

`condor_submit batch_single_empart_guns_tracks_v77/ele_flat2to100_PU0/batch_harvest.sub`
