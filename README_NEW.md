# ntuple-analysis

PYTHON framework for the analysis of [ROOT](https://root.cern/) `TTree` data using [uproot](https://uproot.readthedocs.io/en/latest/) for the IO and [awkward-array](https://awkward-array.org/doc/main/) for the columnar data analysis.

The tool is developed for the analysis of [FastPUPPI](https://github.com/p2l1pfp/FastPUPPI) but should work with any kind of flat ntuples.

## In this README

- [Features](#features)
- [Requirements](#requirements)
- [Usage](#usage)
  - [First time setup](#first-time-setup)
  - [After first time setup](#after-first-time-setup)
- [Main script](#main-script)
  - [How does the analyze n tuples work?](#how-does-the-analyze-n-tuples-work)
  - [Configuration files for the analyze n tuples script](#configuration-files-for-the-analyze-n-tuples-script)
- [Submitting to the batch system](#submitting-to-the-batch-system)
- [FAQ](#faq)
- [Contributing](#contributing)

## Features

The features

 - To be added

## Requirements

- A computing account on LXPLUS CERN service.
- A CentOS machine, running AlmaLinux release 8.9 (Midnight Oncilla).

## Usage

### First time setup

1. [Clone this repository](https://github.com/cerminar/ntuple-analysis.git) 

    *Or you can create a fork of this repository if you plan to contribute*

2. Log in to a LXPLUS machine.

    *A computing account of CERN is required*
    
    * This step is LXPLUS specific, giving access to a more recent Python and Root version. Edit/skip it accordingly for your specific system.*

    ```
    ssh lxplus.cern.ch
    ```
    
    *After that, create a new directory to store the ntuple-analysis project:*

    ```
    mkdir "your directory"
    ```

    *CD into your newly created directory:*
    ```
    cd "your directory"
    ```

    *Again, perform CD into the ntuple-analysis:*
    ```
    cd ntuple-analysis
    ```

3. [Clone this repository](https://github.com/cerminar/plot-drawing-tools.git) - for Jupyter Notebooks support.


4. Edit the ```setVirtualEnvWrapper.sh``` script to add the HOME directory of your user.

    *You can use nano/vim or your other favorite editor.*
    ```
    vim setVirtualEnvWrapper.sh
    ```

    *Edit the first line of the ```setVirtualEnvWrapper.sh``` script:*
    ```
    export WORKON_HOME=/data/YOUR_USERNAME/.virtualenvs
    ```
    *where the YOUR_USERNAME is your username.*


5. Run a shell script ```setup_lxplus.sh``` to set-up the LXPLUS service.

    ```
    source setup_lxplus.sh
    ```

6. Run another shell script ```setVirtualEnvWrapper.sh``` to initialize virtual environment wrapper. 

     ```
    source setVirtualEnvWrapper.sh
    ```   

    *To learn more about Virtual Environment wrapper, you can take a look at the docs [link](https://virtualenvwrapper.readthedocs.io/en/latest/install.html).*
    
7. Create a virtual environment for the project.

    ```
    mkvirtualenv --system-site-packages -p `which python3.9` -r requirements.txt <venvname>
    ```
    *where venvname is the name of your new virtual environment**

    *If you created a virtual environment in a different way, you can use:*
    ```
    pip install -r requirements.txt
    ```
    *NOTE: python > 3.9 is a requirement.**
    
8. Activate the virtual environment (if it's not active already).

    ```
    workon <venvname>
    ```
    *where venvname is the name of your new virtual environment**

9. In order to use Jupyter Notebooks, we need to reinstall ```traitlets``` package.
    
    ```
    pip uninstall traitlets
    ```

    *and then*
    ```
    pip install traitlets==5.9.0
    ```

10. Install a custom kernel with all of the packages from your virtual environment.

    ```
    python3 -m ipykernel install --name <venvname> --user
    ```
    *where venvname is the name of your new virtual environment**

    *Source: [here](https://stackoverflow.com/questions/28831854/how-do-i-add-python3-kernel-to-jupyter-ipython ))*

11. Launch Jupyter Notebook.
    
    *You can launch it in the LXPLUS service:*
    ```
    jupyter notebook
    ```
    *or, if you are using Windows, to access it from the Windows:*
    ```
    jupyter notebook --no-browser --port=8095
    ```
 
### After first time setup

1. Log in to a LXPLUS machine.

    *A computing account of CERN is required*

    ```
    ssh lxplus.cern.ch
    ```
    
    *CD into the root directory of the ntuple-analysis*
    ```
    cd "your directory"
    ```
2. Run a shell script ```setup_lxplus.sh``` to set-up the LXPLUS service.

    ```
    source setup_lxplus.sh
    ```

3. Run another shell script ```setVirtualEnvWrapper.sh``` to initialize virtual environment wrapper. 

     ```
    source setVirtualEnvWrapper.sh
    ```   

    *To learn more about Virtual Environment wrapper, you can take a look at the docs [link](https://virtualenvwrapper.readthedocs.io/en/latest/install.html).*

4. Activate the virtual environment (if it's not active already).

    ```
    workon <venvname>
    ```
    *where venvname is the name of your existing virtual environment from the first set-up**

    *also, ```lsvirtualenv``` is your friend if you forgot the name of the virtualenv.*

## Main script

The main script is ```analyzeNtuples.py```:

```
python analyzeNtuples.py --help
```

An example of how to run it:
```
python analyzeNtuples.py -f cfg/hgctps.yaml -i cfg/datasets/ntp_v81.yaml -c tps -s doubleele_flat1to100_PU200 -n 1000 -d 0
```

### How does the analyzeNtuples.py work?

Data are read in `collections` of objects corresponding to an `array` and are 
processed by `plotters` which creates set of histograms for different `selections` of the data `collections`.

### Configuration files for the analyzeNtuples.py script

The configuration is handled by 2 YAML files. 

The first YAML (e.g: ```hgctps.yaml```) file specifies:  
   - output directories
   - versioning of the plots
   - collections of samples, i.e. group of samples to be processed homogeneously: for each collection the list of plotters (see below) to be run is provided.

The second YAML files (e.g ```ntp_v81.yaml```) provides:
   - details of the input samples (location of the ntuple files)

Example of the YAML configuration files can be found:
 - [cfg/egplots.yaml](cfg/egplots.yaml)
 - [cfg/datasets/ntp_v92.yaml](cfg/datasets/ntp_v92.yaml)

## Submitting to the batch system

Note that the script ```analyzeNtuples.py``` can be used to submit the jobs to the HTCondor batch system invoking the `-b` option. 
A dag configuration is created and you can actually submit it following the script output.

### Note about HADD job.
For each sample injected in the batch system a DAG is created. The DAG will submitt an `hadd` command once all the jobs will succeed.
However, if you don't want to wait (or you don't care) you can submit also a condor job that will run hadd periodically thus reducing dramatically the latency.
For example:

```condor_submit batch_single_empart_guns_tracks_v77/ele_flat2to100_PU0/batch_harvest.sub```


## FAQ

#### - How can you read ntuple branches or create derived branches?

The list of branches to be read and converted to `Awkward Arrays` format is specified in the module

[collections](python/collections.py)

Instantiating an object of class `DFCollection`. What is actually read event by event depends anyhow on which plotters are actually instantiated (collections are read on-demand).

#### - How can you select a subset of object collection?
Selections are defined as strings in the module:

[selections](python/selections.py)

Different collections are defined for different objects and/or different purposes. The selections have a `name` whcih is used for the histogram naming (see below). Selections are used by the plotters.
Selections can be combined and retrieved via regular expressions in the configuration of the plotters.

### - How can you add a new plotter?
The actual functionality of accessing the objects, filtering them according to the `selections` and filling `histograms` is provided by the plotter classes defined in the module:

[plotters](python/plotters.py)

Basic plotters are already available, most likely you just need to instantiate one of them (or a collection of them) using the `DFCollection` instance you are interested in.
Which collection is run for which sample is steered by the configuration file.

The plotters access one or more collections, select them in several different ways, book and fill the histograms (see below).

### - How can you add a new histogram?
Histograms are handled in the module:

[l1THistos](python/l1THistos.py)

There are different classes of histograms depending on the input object and on the purpose.
To add a new histogram to an existing class it is enough to add it in the corresponding constructor and in the `fill` module. The writing of the histos to files is handled transparently.

The histogram naming follows the convention:
`<ObjectName>_<SelectionName>_<GenSelectionName>_<HistoName>`

This is assumed in all the `plotters` and in the code to actually draw the histograms.

#### Histogram drawing

Of course you can use your favorite set of tools. 
I use mine [plot-drawing-tools](https://github.com/cerminar/plot-drawing-tools), 
which is based on `jupyter notebooks`.

```
cd ntuple-analysis
git clone git@github.com:cerminar/plot-drawing-tools.git
jupyter-notebook
```

## Contributing

If you want to contribute to this project, you are very welcome, just fork the project, set-up it on your own machine 
and play with it. If you have any questions, post it on the issues/discussions tab.

Currently, I can't figure out how to do some manipulation using the `awkward array` or `uproot`....you can take a look at examples and play witht the arrays in:
[plot-drawing-tools/blob/master/eventloop-uproot-ak.ipynb](https://github.com/cerminar/plot-drawing-tools/blob/master/eventloop-uproot-ak.ipynb)
