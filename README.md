# SURFsara-PTC-Python-Parallel-and-GPU-Programming

This repository holds the training material used in the PRACE PTC training at SURFsara entitled: Python Parallel and GPU Programming.

For the parallel and GPU sessions, we will use the [Cartesius machines](https://userinfo.surfsara.nl/systems/cartesius) that are generously made available by SURFsara for the OBELICS school. You will need to connect to them through the ssh protocol (natively installed on Linux and Mac).

For Windows users, we recommend these tools to connect via ssh:
- [Putty](http://www.putty.org/) & [Winscp](https://winscp.net)
- Or, on Windows 10: use the [native bash environment](https://msdn.microsoft.com/en-us/commandline/wsl/install_guide)

## Connect to Cartesius

To connect to Cartesius, please open a terminal and use the following command:

    ssh ptcXXX@cartesius.surfsara.nl

Where XXX is the account number you received via email. Type in the password you received together with your login name and press enter.
NOTE that, the cursor won't move while tying in the password, this is normal, so just keep typing.

The account is made available for duration of the course. 

## Preparation for Hands-on session of GPU programming

Login to Cartesius, clone the git repository and generate a key pair.

    ssh ptcXXX@cartesius.surfsara.nl
    git clone https://github.com/vcodreanu/SURFsara-PTC-Python-Parallel-and-GPU-Programming.git
    
    ssh-keygen -t rsa

Press Enter three times. A key pair will be generated for you in directory .ssh. Copy the contents of the public key to file authorized_keys:

    cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
    chmod 700 ~/.ssh
    chmod 600 ~/.ssh/authorized_keys

## Submit a job to Cartesius
When the hands-on session starts, submit the following job to Cartesius:

    cd SURFsara-PTC-Python-Parallel-and-GPU-Programming/gpu_programming
    sbatch job.jupyter.gpu

Open a new terminal and do the following:

    ssh -L5XXX:localhost:5XXX accntXXX@vis.cartesius.surfsara.nl

Note that, you need to replace XXX with the three digits of your login account.
   
Use this command to check your job:

   squeue –u $(whoami)
   
To cancel your job:

   scancel JobID
   
If your job is running, you can open your browser and go to localhost:5XXX (replace XXX with the three digits of your own account). You should be able to see the Jupyter notebook now.
