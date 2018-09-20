# ntuple-tools

The python scripts in this repository should help you get started analysing the [HGCal ntuple](https://github.com/CMS-HGCAL/reco-ntuples) and/or the [HGCAL L1 TP ntuples](https://github.com/PFCal-dev/cmssw/tree/hgc-tpg-devel-CMSSW_10_3_0_pre4/L1Trigger/L1THGCal/plugins/ntuples)

## Pre-requisites




Setup a `virtualenv` using `virtualenvwrapper`.

Follow the `virtualenvwrapper` [installation instructions](https://virtualenvwrapper.readthedocs.io/en/latest/install.html) to install it in the `~/.local/` directory (using `$ pip install --user virtualenvwrapper`). This needs to be done only once for your account.

For setting up the python version on lxplus you can just source the script:

`source setup_lxplus.sh`

For starting using virtualenvwrapper

`source setVirtualEnvWrapper.sh`

The first time you will have to create the actual instance of the `virtualenv`:

`mkvirtualenv <venvname>`

The requirements for the virtualenv setup are in are in the file:

[requirements.txt](requirements.txt)

You can use the file directly using:

`pip install -r requirements.txt`

After this initial (once in a time) setup is done you can just activate the virtualenv calling:

`workon  <venvname>`

(`lsvirtualenv` is your friend in case you forgot the name).


### HGCAL L1 TPG analysis

`python analyzeHgcalL1Tntuple.py -f samples.cfg -c test_hadGUNs -s all -d 4`

The configuration of the available ntuples and of the other parameters is maintained in the file:

[samples.cfg](samples.cfg)

Note that the script can also be used to submit the analysis on the CERN HTCondor batch system

`python analyzeHgcalL1Tntuple.py --help`

for the details.
