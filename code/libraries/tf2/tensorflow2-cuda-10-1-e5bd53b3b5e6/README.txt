1) Open a terminal and clone this repository in your scratch space.

     cd /cs/scratch/<username>
     hg clone https://sysc-public.hg.cs.st-andrews.ac.uk/tensorflow2-cuda-10.1
     cd tensorflow2-cuda-10.1

   Where <username> is your CS username.

2) Edit the file requirements.txt to add any extra modules you might need.

3) Run the following command to create the virtual environment (called venv):

     sh build.sh

   This only needs to be done once (unless you wish to delete and re-create the virtual environment).

4) Run the following command to activate the virtual environment:

     source venv/bin/activate

   This needs to be done once per terminal session.

5) Run the following command to Check that it's all working.

     python hello_tensorflow.py
