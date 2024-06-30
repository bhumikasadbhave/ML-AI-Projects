Protein Sequence Classification
------------------------------------------------------
(Jupyter Notebook is to be run sequentially)


Dataset: Protein Sequences (sequences.fasta)

Goal: Analysis and clustering of protein sequences

Tasks Performed:

Part 1: Data Import
- Loading the protein sequence dataset in the .fasta format.
- The notebook employs the `biopython` package to read protein sequences from a `.fasta` file. 
- The `SeqIO` module is used to parse the file and extract sequences along with their metadata.
- A function is defined to load the sequences from a specified path.
- Displaying Sequence Information: The first few sequences are iterated over and printed to display their ID, description, sequence, and length.

Part 2: Sequence Alignment
- Using Smith-Waterman Algorithm Uto calculate an alignment score between two sequences
- This part involves calculating the alignment score between two sequences using the Smith-Waterman algorithm.
- The `pairwise2.align.localms` function from `biopython` is utilized for this purpose.
- The alignment score is computed based on match scores and gap penalties, which are crucial for sequence compatibility.

Part 3: Sequence Embedding
- Embedding in Euclidean Vector Space
- Protein sequences are embedded into a 100-dimensional Euclidean vector space.
- This embedding is necessary for applying clustering algorithms.

Part 4: Clustering
- Affinity Propagation to cluster the embedded sequences.
- Affinity Propagation is a clustering algorithm that identifies exemplars among the data points and forms clusters based on their similarities.

Part 5: Evaluation
- The clustering results are evaluated using Normalized Mutual Information (NMI).
- NMI measures the similarity between the true labels and the predicted clusters.

Part 6: Visualization
- A proximity matrix is visualized to show the similarity between different sequences.
- The matrix helps in understanding the clustering and the relationships between sequences.




