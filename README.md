# Quantum-Graph-Spectra

Final project for CS 269Q and launchpad for further investigations into the application of quantum computing to spectral graph theory. Here, we leverage the Variational Quantum Eigensolver (VQE) algorithm to efficiently compute eigenvalues of adjacency and Laplacian (normalized or otherwise) matrices of graphs.

### Prerequisites

* python3
* pyQuil (instructions: https://github.com/rigetti/pyquil)
* numpy
* scipy
* matplotlib
* networkx

## Running tests

This algorithm may be run using the Quantum Virtual Machine. To do this, run `$ qvm -S` and `$ quilc -S`. Then make sure the line in vqe.py `qc = get_qc("9q-generic-qvm")` is uncommented and any other `get_qc` is commented out.

It may also be run on QCS (invitation to join required--this can be requested on their site). Transfer the files to the remote host following their instructions, set up the virtual enviroment, and you can run the tests as done locally. Make sure the line in vqe.py `qc = get_qc(<your lattice>)` is uncommented and any other `get_qc` is commented out.

For both methods, tests may be run using `$ python3 performance.py`, after having put the desired print statements for testing at the bottom of the file.

## Contributing

1. Fork it!
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request :)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Rigetti Computing, for their generosity in giving us credits for testing on QCS
* Will Zeng and Dan Boneh, for teaching the class CS 269Q - Elements of Quantum Computer Programming
* Dulinger Ibeling and Jon Braatz, for TA-ing CS 269Q
* Jacob Fox, Aaron Sidford, and Erik Bates for their thoughts and suggestions