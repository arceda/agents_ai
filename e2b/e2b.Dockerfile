FROM python:3

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir numpy scipy pandas matplotlib scikit-learn seaborn 
    # Note: we had to merge the two "pip install" package lists here, otherwise
    # the last "pip install" command in the OP may break dependency resolution…

CMD ["cat", "/etc/os-release"]

