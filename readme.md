This collection of scripts assembles and phases the spontaneous rocking data collected at NanoMAX during

* Experiment 20180278 Jun 2019 (`data1`) and
* Follow-up in-house time 20200322 (`data2`).

The analysis is described in
> A. Björling et al, "Three-dimensional coherent Bragg imaging of spontaneously rotating nanoparticles", in preparation (2020).

The data are deposited separately in a root folder referred to as PATH. There are nine particle hits analyzed in total four in `data1` and five in `data2`.

Each particle hit is analyzed in its own folder, where the following scripts are run in sequence.

* `1_pick_frames.py` picks a specific CDI hita and crops the data.
* `2_assemble.py` does the diffraction volume assembly.
* `3_prepare_pynx.py` crops and scales the model volume for phasing.
* `4_reconstruct` or `4_reconstructr.sbatch` does the phasing with PyNX
* `5_validate and rectify.py` resamples the particle on an orthogonal grid and does the FSV analysis.

These scripts depend on the [bcdi-assemble](https://github.com/maxiv-science/bcdi-assemble) library for assembly, and on [PyNX](http://ftp.esrf.fr/pub/scisoft/PyNX/) for phasing.
