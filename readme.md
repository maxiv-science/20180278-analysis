This collection of scripts assembles and phases the spontaneous rocking data collected at NanoMAX during

* Experiment 20180278 Jun 2019 (`data1`) and
* Follow-up in-house time 20200322 (`data2`).

The analysis is described in
> A. Björling et al, "Three-dimensional coherent Bragg imaging of spontaneously rotating nanoparticles", in preparation (2020).

The data are deposited separately in a root folder referred to as DPATH (supplied to the first script as the only command-line argument). There are nine particle hits analyzed in total four in `DPATH/data1` and five in `DPATH/data2`.

Each particle hit is analyzed in its own folder, where the following scripts are run in sequence.

* `1_pick_frames.py $DPATH` picks a specific CDI hit and crops the data.
* `2_assemble.py` does the diffraction volume assembly.
* `3_prepare_pynx.py` crops and scales the model volume for phasing.
* `4_reconstruct` or `4_reconstructr.sbatch` does the phasing with PyNX
* `5_validate and rectify.py` resamples the particle on an orthogonal grid and does the FSV analysis.

These scripts depend on the [bcdi-assemble](https://github.com/maxiv-science/bcdi-assemble) library for assembly, and on [PyNX](http://ftp.esrf.fr/pub/scisoft/PyNX/) for phasing.
