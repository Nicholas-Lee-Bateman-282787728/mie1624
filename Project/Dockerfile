FROM python:3

#ADD /usr/local/dataset /usr/local/
COPY sample_submission.py /usr/src/
#COPY train /usr/local/


RUN pip install numpy>=1.15.4
RUN pip install matplotlib>=3.0.1
RUN pip install pandas>=0.24.2 
RUN pip install scipy
RUN pip install Cython
RUN pip install scikit-learn 

CMD [ "python3", "/usr/src/sample_submission.py" ]
