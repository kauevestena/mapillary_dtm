# steps for ubuntu/debian, cloning at home:
# you can efortlessly create a venv using vscode, OR:
    git clone https://github.com/kauevestena/mapillary_dtm.git

    cd mapillary_dtm

    <python_exec_path> -m venv --without-pip $HOME/mapillary_dtm/.venv

    source .venv/bin/activate

# and optionally (to ignore all venv files on the management tree):

    echo "*" > .venv/.gitignore

# then install dependencies:

    $HOME/mapillary_dtm/.venv/bin/python -m pip install -r requeriments.txt


