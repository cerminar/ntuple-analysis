# ntuple-analysis

PYTHON framework for the analysis of [ROOT](https://root.cern/) `TTree` data using [uproot](https://uproot.readthedocs.io/en/latest/) for the IO and [awkward-array](https://awkward-array.org/doc/main/) for the columnar data analysis.

The tool is developed for the analysis of [FastPUPPI](https://github.com/p2l1pfp/FastPUPPI) but should work with any kind of flat ntuples.

## Pre-requisites: first time setup

The tool can be run on any private machines using just `python`, `pip` and `venv`.
For convenience, the procedure to manage venvs using  `virtualenvwrapper` is described.
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

`python analyzeNtuples.py -f cfg/egvalid.yaml -i cfg/datasets/ntpfp_131Xv3.yaml -p egmenu -s doubleele_flat1to100_PU200 -n 1000 -d 0`

## General idea

The analysis is defined by a `yaml` file and a `python` module of the same name. They define a number of collection of plotters which read some data and fill a set of plots for a list of data selections. In case gen matching is needed the same plots are filled for all the combinations of data and gen selections specified in the configuration.
One of the collection is specified via command line arguments (`-p` option).


Data are represented by `collections` of objects which can be read from `ROOT::TTree` files or filled on the fly. They are processed by `plotters` which creates set of histograms for different `selections` of the data `collections`.

The `plotters`, the `histograms` and the various `selections` are defined in the configuration `python` file.

The ouput histograms are saved in the output file following a naming convention:

`<Histo class name>/<collection name>_<selection name>_<histo name>`

In case gen matching is performed the naming convention becomes:

`<Histo class name>/<collection name>_<selection name>_<gen collection name>_<gen selection name>_<histo name>`

The histogram classes handle saving and reading histograms to/from file transparently.


### Configuration file
The configuration is handled by 2 yaml files.

One specifying
   - output directories
   - versioning of the plots
   - collections of samples, i.e. group of samples to be processed homogeneously: for each collection the list of plotters (see below) to be run is provided.

The other prividing
   - details of the input samples (location of the ntuple files)

Example of configuration file can be found in:
 - [cfg/eg_genmatch.yaml](cfg/eg_genmatch.yaml)
 - [cfg/datasets/ntpfp_131Xv3.yaml](cfg/datasets/ntpfp_131Xv3.yaml)

So you can run the same set of plotters on different input ntuples.

### Reading ntuple branches or creating derived ones

The list of branches to be read and converted to `Awkward Arrays` format is specified in the module

[fastpuppi_collections.py](cfg/datasets/fastpuppi_collections.py)

Instantiating objects of class `DFCollection`. What is actually read event by event depends anyhow on which plotters are actually instantiated (collections are read on-demand).
For each collection, a function adding columns beyond those in the root file can be defined.

New collections can be created for example combining those read from the root file.

### Selecting subsets of object collections
Selections are defined as strings in the module:

[selections](python/selections.py)

Different collections are defined for different objects and/or different purposes. The selections have a `name` which is used for the histogram naming (see below). Selections are used by the plotters.
Selections can be combined and retrieved via regular expressions in the configuration of the plotters.

### Adding a new plotter
The actual functionality of accessing the objects, filtering them according to the `selections` and filling `histograms` is provided by the plotter classes. The base ones are defined in the module:

[plotters](python/plotters.py)

Basic plotters are already available, most likely you just need to instantiate one of them (or a collection of them) using the `DFCollection` instance you are interested in.
Which collection is run for which sample is steered by the configuration file.

The plotters access one or more collections, select them in several different ways, book and fill the histograms (see below).

### Adding a new histogram
Histograms are handled in the module:

[histos](python/histos.py)

There are different classes of histograms depending on the input object and on the purpose.
To add a new histogram to an existing class it is enough to add it in the corresponding constructor and in the `fill` module. The writing of the histos to files is handled transparently.

The histogram naming follows the convention:
`<ObjectName>_<SelectionName>_<GenSelectionName>_<HistoName>`

This is assumed in all the `plotters` and in the code to actually draw the histograms.

Histograms are coputed using boost histograms via the [hist](https://hist.readthedocs.io/en/latest/user-guide/notebooks/Histogram.html) but saved as ROOT histograms.

## Histogram drawing

Of course you can use your favorite set of tools: histograms are coputed using boost histograms via the [hist](https://hist.readthedocs.io/en/latest/user-guide/notebooks/Histogram.html) but saved as ROOT histograms.

The [draw.py](draw.py) script allows to call different drawing modules defined for the various analysis modules. Given a list of files the will retrieve histograms by `collection` and by `selection` name and plot them. The resulting `png` and `pdf` files are saved to the target directory specified via command line.

Additionally, interactive plotting can be done using the `jupyter notebooks` in [plot-drawing-tools](https://github.com/cerminar/plot-drawing-tools).

```
cd ntuple-analysis
git clone git@github.com:cerminar/plot-drawing-tools.git
jupyter-notebook
```

## Examples

- Running GEN matching to compute efficiency on e/g menu objects and draw plots:

```
python  analyzeNtuples.py -f cfg/eg_genmatch.yaml -i cfg/datasets/ntpfp_131Xv3.yaml -p egmenu  -s doubleele_flat1to100_PU200 -n 1000 -d 0

python draw.py -m cfg/eg_genmatch_draw.py -w egmenu_ele --input-files path/file1.root:label1,path/file2.root:label2 --target-dir /Users/cerminar/CERNbox/www/plots/test2/
```

- Runnig rate computations on e/g menu objects:
```
 python  analyzeNtuples.py -f cfg/eg_rate.yaml -i cfg/datasets/ntpfp_131Xv3.yaml -p rate_menu  -s nugun_alleta_pu200 -n 1000 -d 0

 python draw.py -m cfg/eg_rate_draw.py -w menu_rate --input-files plots/histos_nugun_alleta_pu200_ratemenu_v160A.v131Xv1A.root:menu-v31,plots/histos_nugun_alleta_pu200_ratemenu_v160A.131Xv3.root:menu-v33 --target-dir /Users/cerminar/CERNbox/www/plots/test2/
```

- Running GEN matching to compute efficiency on HGC TPs objects:

``` 
python  analyzeNtuples.py -f cfg/hgctps.yaml -i cfg/datasets/ntpfp_v100.yaml -p genmatch  -s doubleele_flat1to100_PU200 -n 1000 -d 0
```

- Running rate computations on HGC clusters
```
python  analyzeNtuples.py -f cfg/hgctps.yaml -i cfg/datasets/ntpfp_v100.yaml -p rate  -s doubleele_flat1to100_PU200 -n 1000 -d 0
```

- Computing HGC cluster occupancies per CTL1 region

```
 python  analyzeNtuples.py -f cfg/l1ct_occupancy.yaml -i cfg/datasets/ntpfp_v100.yaml -p tps  -s doubleele_flat1to100_PU200 -n 1000 -d 0
```



## HELP

I can't figure out how to do some manipulation using the `awkward array` or `uproot`....you can take a look at examples and play witht the arrays in:
[plot-drawing-tools/blob/master/eventloop-uproot-ak.ipynb](https://github.com/cerminar/plot-drawing-tools/blob/master/eventloop-uproot-ak.ipynb)

<!-- ## Submitting to the batch system

Note that the script `analyzeNtuples.py` can be used to submit the jobs to the HTCondor batch system invoking the `-b` option. A dag configuration is created and you can actually submit it following the script output.

### Note about hadd job.
For each sample injected in the batch system a DAG is created. The DAG will submitt an `hadd` command once all the jobs will succeed.
However, if you don't want to wait (or you don't care) you can submit also a condor job that will run hadd periodically thus reducing dramatically the latency.
For example:

`condor_submit batch_single_empart_guns_tracks_v77/ele_flat2to100_PU0/batch_harvest.sub` -->
