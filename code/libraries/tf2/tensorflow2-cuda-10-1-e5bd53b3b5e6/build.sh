#!/bin/bash

# Uncomment to log all commands as they run
#set -x

PYTHON=/usr/local/python/bin/python3.7
CUDA_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
VENV_PATH=${PWD}/venv

[ ! -e ${VENV_PATH} ] || { echo "Path ${VENV_PATH} already exists (remove or rename it and run script again)"; exit 1; }
${PYTHON} -m venv ${VENV_PATH} || { echo "error creating virtual environment" ; exit 1; }
echo "export LD_LIBRARY_PATH=${CUDA_LIBRARY_PATH}" >> ${VENV_PATH}/bin/activate

source ${VENV_PATH}/bin/activate
pip install --upgrade pip || { echo "error upgrading pip"; exit 1; }
pip install --upgrade setuptools || { echo "error installing setuptools"; exit 1; }
pip install --requirement requirements.txt || { echo "error installing requirements"; exit 1; }

echo
echo "            VENV path: ${VENV_PATH}"
echo "  Python command path: ${PYTHON}"
echo "    CUDA library path: ${CUDA_LIBRARY_PATH}"
echo
echo "Type the following in a terminal to activate the virtual environment:"
echo
echo "  source ${VENV_PATH}/bin/activate"
echo
