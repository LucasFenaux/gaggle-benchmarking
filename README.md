# gaggle_experiments
Code to produce the evaluation of the gaggle package!

To clone the submodules. Either append ```--recurse-submodules``` to the regular clone command.

Alternatively you can cd into the submodules and run ```git submodule update --init --recursive```


To run the experiments and get the plots presented in the paper, follow the following instructions:

First install the dependencies:

<pre><code>conda env create -f environment.yml</code></pre>

Then for each of the experiments (mnist_lenet, cartpole and rastrigin), go into their respective directory and run:

<pre><code>bash run.sh</code></pre>

This will save all the relevant and necessary pickle files into a newly created "Results" directory (again within each
experiment folder).

The cells in the ```plot_results.ipynb``` ipython notebooks within each directory can then be run to produce the 
desired plots. The plots will also be saved in the respective Results directories.