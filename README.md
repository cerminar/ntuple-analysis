# ntuple-analysis

PYTHON framework for the analysis of [ROOT](https://root.cern/) `TTree` data using [uproot](https://uproot.readthedocs.io/en/latest/) for the IO and [awkward-array](https://awkward-array.org/doc/main/) for the columnar data analysis.

The tool is developed for the analysis of [FastPUPPI](https://github.com/p2l1pfp/FastPUPPI) but should work with any kind of flat ntuples.

## Pre-requisites: first time setup

The tool can be run on any private machines using just `python`, `pip` and `virtualenvwrapper`.
If you plan to run it on lxplus you might want to look at the point `1` below.

### 1. lxplus setup

This step is `lxplus` specific, givin access to a more recent `python` and `root` version.
Edit/skip it accordingly for your specific system.

`source setup_lxplus.sh`

### 2. install `virtualenvwrapper`

This stetp needs to be done **only once** for your account and can be done with whatever `python` version is in use in the system.

For some reason the current `CMSSW` scrips seems to deliver an inconsistent setup of `virtualenv` and `virtualenvwrapper`, for this reason we force a new installation in `~/.local` using:

`pip install --ignore-installed --user virtualenv==15.1.0 virtualenvwrapper`

For a more complete overview of the procedure you can refer to
`virtualenvwrapper` [installation instructions](https://virtualenvwrapper.readthedocs.io/en/latest/install.html)

### 3. setup `virtualenvwrapper`

For starting using virtualenvwrapper

`source setVirtualEnvWrapper.sh`

### 4. create a virtualenv for the project

The **first time** you will have to create the actual instance of the `virtualenv`:

``mkvirtualenv --system-site-packages
 -p `which python3.9` -r requirements.txt <venvname>``

[requirements.txt](requirements.txt)

You can use the file directly using for example:

`pip install -r requirements.txt`

*NOTE*: `python > 3.9` is a requirement.


## Setup after first installation

### 1. lxplus setup

This step is `lxplus` specific, givin access to a more recent `python` and `root` version.
Edit/skip it accordingly for your specific system.

`source setup_lxplus.sh`

### 2. setup `virtualenvwrapper`

For starting using virtualenvwrapper

`source setVirtualEnvWrapper.sh`

### 3. activate the `virtualenv`

After this initial (once in a time) setup is done you can just activate the virtualenv calling:

`workon  <venvname>`

(`lsvirtualenv` is your friend in case you forgot the name).


### Conda environment
You can use also conda to install all the dependencies and root

```bash
conda create env_name python=3.11
conda activate env_name
conda install root              #In the conda-forge channel
pip install -r requirements.txt
```


## Running the analysis

The main script is `analyzeNtuples.py`:

`python analyzeNtuples.py --help`

An example of how to run it:

`python analyzeNtuples.py -f cfg/hgctps.yaml -i cfg/datasets/ntp_v81.yaml -c tps -s doubleele_flat1to100_PU200 -n 1000 -d 0`

## General idea

Data are read in `collections` of objects corresponding to an `array` and are processed by `plotters` which creates set of histograms for different `selections` of the data `collections`.


### Configuration file
The configuration is handled by 2 yaml files.

One specifying
   - output directories
   - versioning of the plots
   - collections of samples, i.e. group of samples to be processed homogeneously: for each collection the list of plotters (see below) to be run is provided.

The other prividing
   - details of the input samples (location of the ntuple files)

Example of configuration file can be found in:
 - [cfg/egplots.yaml](cfg/egplots.yaml)
 - [cfg/datasets/ntp_v92.yaml](cfg/datasets/ntp_v92.yaml)


### Reading ntuple branches or creating derived ones

The list of branches to be read and converted to `Awkward Arrays` format is specified in the module

[collections](python/collections.py)

Instantiating an object of class `DFCollection`. What is actually read event by event depends anyhow on which plotters are actually instantiated (collections are read on-demand).

### Selecting subsets of object collections
Selections are defined as strings in the module:

[selections](python/selections.py)

Different collections are defined for different objects and/or different purposes. The selections have a `name` whcih is used for the histogram naming (see below). Selections are used by the plotters.
Selections can be combined and retrieved via regular expressions in the configuration of the plotters.

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


## Histogram drawing

Of course you can use your favorite set of tools. I use mine [plot-drawing-tools](https://github.com/cerminar/plot-drawing-tools), which is based on `jupyter notebooks`.

`cd ntuple-analysis`
`git clone git@github.com:cerminar/plot-drawing-tools.git`
`jupyter-notebook`

## HELP

I can't figure out how to do some manipulation using the `awkward array` or `uproot`....you can take a look at examples and play witht the arrays in:
[plot-drawing-tools/blob/master/eventloop-uproot-ak.ipynb](https://github.com/cerminar/plot-drawing-tools/blob/master/eventloop-uproot-ak.ipynb)

## Submitting to the batch system

Note that the script `analyzeNtuples.py` can be used to submit the jobs to the HTCondor batch system invoking the `-b` option. A dag configuration is created and you can actually submit it following the script output.

### Note about hadd job.
For each sample injected in the batch system a DAG is created. The DAG will submitt an `hadd` command once all the jobs will succeed.
However, if you don't want to wait (or you don't care) you can submit also a condor job that will run hadd periodically thus reducing dramatically the latency.
For example:

`condor_submit batch_single_empart_guns_tracks_v77/ele_flat2to100_PU0/batch_harvest.sub`
