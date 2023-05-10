# Gaggle Experiments
Code to produce the evaluation of the gaggle package!

To clone the submodules. Either append ```--recurse-submodules``` to the regular clone command.

Alternatively you can cd into the submodules and run ```git submodule update --init --recursive```


To run the experiments and get the plots presented in the paper, follow the following instructions:

First install the dependencies:

<pre><code>pip install -r requirements.txt</code></pre>

Then for each of the experiments (mnist_lenet, cartpole and rastrigin), go into their respective directory and run:

<pre><code>bash run.sh</code></pre>

This will save all the relevant and necessary pickle files into a newly created "Results" directory (again within each experiment folder).

The cells in the ```plot_results.ipynb``` ipython notebooks within each directory can then be run to produce the desired plots. The plots will also be saved in the respective Results directories.


We included the code we used for the GitHub repositories of the papers we compared against. Their code can also be found here:

[Pygad](https://github.com/ahmedfgad/GeneticAlgorithmPython)

[TorchGA](https://github.com/ahmedfgad/TorchGA)

[LEAP](https://github.com/AureumChaos/LEAP)