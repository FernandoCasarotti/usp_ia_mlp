# usp_ia_mlp
Python script to solve character recognition, XOR/AND/OR problems and political profile identification with MLP NN algorithm,
following specific datasources for this matter.

In general lines, it works in this way (and you always can use the --h or -help command for better undestanding):
- All the actions can be called from the command line, with the obvious prefix of 'python3 ia_2nd_delivery.py':
    - votes
        - This one runs the algorithm with the default settings (14 hidden nodes, 1000 epochs, finding the best lr)
    - votes_grade
        - This one runs the grid analysis with the data aligned, exploring all 60 combinations (obviously, it takes time to run)
    - votes_chosen_es
        - This version takes the best combination (so far, it is the combination of number 25, with n = 40, lr = 1.0, epochs = 50) and executes the algorithm with that applying the early stop
    - votes_chosen
        - This version takes the best combination (so far, it is the combination of number 25, with n = 40, lr = 1.0, epochs = 50) and runs the algorithm without stopping early
- Explaining how it works
    - For grade
        - It generates the possible analyzes to be done with Keras and places the data in the files. All 60 analyzes, including the graphs, are placed in the file to be able to compare them, so you can really check if what I found is the best
    - For the others
        - It generates the files only with the data of the algorithm in question, and plots the graphs on the screen
- What is not yet automatical
    - The choice of the best grid algorithm, obviously. This has to be analyzed and compared individually and find the best
    - If you find a better one than I found, what you need to do
        - Move on line 297, changing the patience to the one that best suits
        - Move on line 490, setting the parameters of the chosen method
