#FROM centos:centos7


from tensorflow/tensorflow:2.4.2-gpu
LABEL maintainer="Fatima Sadiq"
RUN apt update
RUN apt install git python3-pip -y
COPY . /
WORKDIR /
# download the transformers repo and checkout at specific commit
RUN git clone https://github.com/huggingface/transformers.git && \
 cd transformers && \
 git checkout 5b1b5635d3bd4e24b3c4bb1df4fa48e9ccf5f867
# as per the GitHub page said: install the transformers
RUN cd transformers && pip3 install . --no-cache-dir
# install the dependencies for language modeling
RUN cd transformers/examples/pytorch/language-modeling && pip3 install -r requirements.txt --no-cache-dir
# install the requirements in your directory
RUN pip3 install -r requirements.txt --no-cache-dir
RUN mkdir model
RUN python3 first.py
RUN python3 transformers/examples/pytorch/language-modeling/run_clm.py \
 --model_type gpt2\
 --model_name_or_path gpt2\
 --train_file "./train_tmp.txt"\
 --do_train\
 --validation_file "./eval_tmp.txt"\
 --do_eval\
 --per_gpu_train_batch_size 2\
 --save_steps -1\
 --num_train_epochs 5\
# --fp16\
 --output_dir="./model"
RUN python3 second.py
ENTRYPOINT [ "sh" ]
CMD [ "ls" ]



