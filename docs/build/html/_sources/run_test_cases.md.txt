# Test Cases

In this document, we make an overview of all the capabilities of the library. 
So, we start recalling that we have three packages in the `src` folder:
1. `data`: for static data generation and post-processing;
2. `surrogate`: for the surrogate models;
3. `multi_physics`: for the electro-mechanical solver.

> Consult `report.pdf` in the main directory for a detailed background explanation, test cases, and results, including images and videos.

## Overall Outline

We first describe how the data and surrogate packages are used to train the surrogate models. We then present and validate the multiphysics solver, and finally assess its performance when integrating the pre-trained surrogate models.

## Surrogates

We proceed with the following structure in mind:
1. use the data generation pipeline to produce an example of a dataset (smaller compared to the one we used to train the models provided for the following tests);
2. evaluate the performance of the models we trained on the new dataset;
3. show the training pipeline by building a new (weak) model on the new (small) dataset;

### Test 1. Cantilevered beam

> In this case the upper plate is clamped at the left end and free at the right one (i.e. it's a cantilever).

First you can create the __dataset__. Some plots of the domain and the (traditional) numerical solutions corresponding a combination of parameters present in the dataset you've just generated.
If you want you can play with the number of __workers__ to make the generation faster. 

```bash
python -m src.data.generate --folder "test/test1" --data_file "test/test1.csv" --geometry_input "geometries/cantilever1.geo" --workers 2
```

> Note the real bottleneck of the process in this case is the mesh generation section, since the FOM (Full Order Model) is quite fast, since it's only a laplacian.

At the end of the data generation, a plot of the solution for a specific configuration is produced. Figures from 1 to 5 in `test/img` are an example of the output of this process. 

Then you can __evaluate__ the model for the electrostatic potential we already trained on the test set you generated.
```bash
python -m src.surrogate.evaluate --folder "test/test1" --model_path "models/potential1.keras" --target "potential" 
```

> Since the process data splitting is random, we actually don't know if the model has already seen the data we are generating here. So this is an unbiased estimator of the real performance of the model.

Now you can visualize with some plots the predictions that the model can make.
```bash
python -m src.surrogate.predict --folder "test/test1" --model_path "models/potential1.keras"  --target "potential" 
```

Figures from 10 to 12 are an example of the plots produced by this command.

You can do the same for with the model for the normal derivative of the potential on the boundary of the upper plate.
```bash
python -m src.surrogate.evaluate --folder "test/test1" --model_path "models/derivative1.keras" --target "normal_derivative" 
```
```bash
python -m src.surrogate.predict --folder "test/test1" --model_path "models/derivative1.keras"  --target "normal_derivative" 
```

Figures from 6 to 9 are an example of the plots produced by this command.

If you want you can try training the model, with the dedicated module.
```bash
python -m src.surrogate.train --folder "test/test1" --model_path "models/potential.keras" --target "potential"
```

```bash
python -m src.surrogate.train --folder "test/test1" --model_path "models/derivative.keras" --target "normal_derivative"
```

You can also use a GPU if available, but you must satisfy the CUDA requirements yourself by properly configuring your environment. This setup may not be straightforward and, if done incorrectly, can lead to warnings. For example, you may want to run in the terminal something like:
```bash
mamba activate env-name
mamba install cuda-cudart cuda-version=12 -y
```

Logs of the training are available and you can open them in your browser using tensorboard:
```bash
tensorboard --logdir logs
```

### Test 2: Bigger Deformation

You can just do the same as above but with the following changes:
- use `test2` instead of `test1`;
- changing the reference geometry to `geometries/cantilever2.geo`; 
- consider now the models `models/potential2.keras` and `models/derivative2.keras`.

### Test 3: Clamped-Clamped Beam

You can just do the same as above but with the following changes:
- use `test3` instead of `test1`;
- changing the reference geometry to `geometries/clamped.geo`; 
- consider now the models `models/potential3.keras` and `models/derivative3.keras`.

## Multi-Physics

### Classical Multi-Physics Solver

__Visualization.__ First, you can run the classical multi-physics solver in the case of cantilever with big deformation of the upper plate with
```bash
python -m src.multi_physics.solver \
  --nmodes 4 \
  --template-geo geometries/cantilever2.geo \
  --dt 5e-6 \
  --nsteps 80 \
  --Vdc 0 \
  --Vac 230 \
  --freq 2.5e3 \
  --Vupper 0 \
  --Vouter 0 \
  --omega 6.3e5 3.9e6 1.1e7 2.1e7 \
  --mass 1e-12 1e-12 1e-12 1e-12 \
  --zeta 0.01 0.01 0.01 0.01 \
  --print-every 1 \
  --fail-fast \
  --no-outer-bc \
  --workdir "temp/visualization"
```
Then, you can go to the `temp/visualization` folder and visualize the solution by opening the `temp/visualization/results` subfolder in ParaView. Remember to select the scalar field `phi` (electrostatic potential) for coloring. As an example of what the output looks like, we provide `test/videos/visualization.mp4`, which is the video of `phi` over time.

__Mesh convergence in time.__ You can perform a convergence study with respect to the time step in the standard case with the following command:
```bash
python test/test_convergence.py
```
Results are saved in the `temp/convergence` folder. We provide `test/videos/mesh-convergence-time.mp4` and `test/img/mesh-convergence-time.png` with the plots produced by this test, in the case of a cantilever: the first for the evolution of the displacement and the second for one of the capacitance.

__Test number of modes.__ You can perform a test which compares the results changing the number of modes (ranging from 1 to 4) used to project the mechanical deformation. 
```bash
python test/test_nmodes.py
```
Indeed, you can observe by yourself that the only relevant mode is the first one. Results are saved in the `temp/nmodes` folder. We provide `test/videos/number-of-modes.mp4` and `test/img/number-of-modes.png` with the plots produced by this test, in the case of a cantilever: the first for the evolution of the displacement and the second for one of the capacitance.

__Additional flags.__
For the last two commands you can use the following additional flags:
1. `--no-simulation`: visualize the solution again after running test, you can add the flag;
2. `--save-frames`: save the plot of the capacity over time and the video of the displacement over time;
3. `--big-deformation`: run the test in the case of a cantilever with bigger deformation (this flag works also for the first command of this section);
4. `--clamped`: run the test in the case of clamped-clamped upper plate (this flag works also for the first command of this section).

### DL Multi-Physics Solver

__Visualization.__ You can run the DL multi-physics solver (this time considering realistic deformation scales) using pre-trained (DeepONet) surrogates for the electrostatic part of the physics through the following command:
```bash
python -m src.multi_physics.solver \
  --nmodes 4 \
  --template-geo geometries/cantilever1.geo \
  --dt 1e-5 \
  --nsteps 40 \
  --Vdc 0 \
  --Vac 5 \
  --freq 2.5e3 \
  --Vupper 0 \
  --Vouter 0 \
  --omega 6.3e5 3.9e6 1.1e7 2.1e7 \
  --mass 1e-12 1e-12 1e-12 1e-12 \
  --zeta 0.01 0.01 0.01 0.01 \
  --print-every 1 \
  --fail-fast \
  --derivative-nn-path models/derivative1.keras \
  --potential-nn-path models/potential1.keras \
  --no-outer-bc \
  --workdir "temp/visualization-dl"
```
Then, you can visualize the solution using ParaView as in the previous case. This time you can select between three scalar field: `phi`, `phi_pred` and `phi_error`. Results are saved in the `temp/visualization-dl` folder. As an example of what the output looks like, we provide `test/videos/visualization-dl.mp4`, which is the video of `phi_error` over time.

__Performance.__ You can evaluate the overall performance of the DL-ROM by running:
```bash
python test/test_performance.py
```
This script compares the DL-ROM with the classical ROM, performing a profiling of execution times and an assessment of accuracy. In particular, the evaluation of the latter focuses on the displacement and the capacitance (quantity of interest).
Results are saved in the `temp/performance` folder. We provide `test/videos/performance.mp4` and `and test/img/performance.png` with the plots produced by this test, in the case of cantilever: the first for the evolution of the displacement and the second for one of the capacitance.

__Geometry parameters.__ You can test the robustness of the DL-ROM over a grid of geometric parameters by running:
```bash
python test/test_geometry.py
```
By default, this test compares the DL-ROM with the classical ROM on a 3-by-3 grid, varying the overetch from $0$ to $0.5$ and the distance between the plates from $1.5$ to $2.5$. For every geometry, it evaluates the speedup and the normalized L-infinity errors of the capacitance and displacement, then reports their mean and standard deviation. The `--grid-refinement` option changes the number of values used for each parameter.
Results are saved in the `temp/geometry` folder. For instance, the expected metrics for the case of a cantilever are:
<div align="center">

| Metric | Mean | Std. dev. |
|---|---:|---:|
| Speedup | $19.48$ | $4.465$ |
| $\varepsilon_C$ | $1.572 \times 10^{-2}$ | $6.934 \times 10^{-3}$ |
| $\varepsilon_u$ | $1.822 \times 10^{-2}$ | $9.661 \times 10^{-3}$ |

</div>
where

$$
\varepsilon_C =
\frac{
    \left\|C_{\mathrm{pred}}-C_{\mathrm{ref}}\right\|_{L^\infty_t}
}{
    \max C_{\mathrm{ref}}-\min C_{\mathrm{ref}}
}, \quad 
\varepsilon_u =
\frac{
    \left\|u_{\mathrm{pred}}-u_{\mathrm{ref}}\right\|_{L^\infty_x L^\infty_t}
}{
    \max u_{\mathrm{ref}}-\min u_{\mathrm{ref}}
}.
$$

__Additional flags.__
For the performance and geometry tests, you can use the following additional flags:
1. `--no-simulation`: visualize the solution again after running test, you can add the flag;
2. `--save-frames`: save the plot of the capacity over time and the video of the displacement over time;
3. `--big-deformation`: run the test in the case of a cantilever with bigger deformation (this flag works also for the first command of this section);
4. `--clamped`: run the test in the case of clamped-clamped upper plate (this flag works also for the first command of this section).


### Memory profiling (optional)

You can use [Memray](https://github.com/bloomberg/memray), which is already present in the provided environment, to run a memory inspection and compare the DL-ROM against the classical ROM.

For the classical ROM you can run

```bash
mkdir -p temp/memory && python -m memray run --native -o temp/memory/classical_rom.bin -m src.multi_physics.solver \
  --template-geo geometries/cantilever1.geo \
  --workdir temp/memory/classical_rom_run \
  --nmodes 4 \
  --dt 1e-5 \
  --nsteps 40 \
  --Vdc 0 \
  --Vac 5 \
  --freq 2.5e3 \
  --Vupper 0 \
  --Vouter 0 \
  --omega 6.3e5 3.9e6 1.1e7 2.1e7 \
  --mass 1e-12 1e-12 1e-12 1e-12 \
  --zeta 0.01 0.01 0.01 0.01 \
  --print-every 1 \
  --no-outer-bc \
  --fail-fast
```

and

```bash
python -m memray flamegraph --temporal -o temp/memory/classical_rom.html temp/memory/classical_rom.bin
```

For the DL-ROM you can run

```bash
mkdir -p temp/memory && python -m memray run --native -o temp/memory/dl_rom.bin -m src.multi_physics.solver \
  --template-geo geometries/cantilever1.geo \
  --workdir temp/memory/dl_rom_run \
  --nmodes 4 \
  --dt 1e-5 \
  --nsteps 40 \
  --Vdc 0 \
  --Vac 5 \
  --freq 2.5e3 \
  --Vupper 0 \
  --Vouter 0 \
  --omega 6.3e5 3.9e6 1.1e7 2.1e7 \
  --mass 1e-12 1e-12 1e-12 1e-12 \
  --zeta 0.01 0.01 0.01 0.01 \
  --print-every 1 \
  --fail-fast \
  --no-outer-bc \
  --derivative-nn-path models/derivative1.keras \
  --no-postprocessing
```

and

```bash
python -m memray flamegraph --temporal -o temp/memory/dl_rom.html temp/memory/dl_rom.bin
```

Then you can take a look at the output HTML files.