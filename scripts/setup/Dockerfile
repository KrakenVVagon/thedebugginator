FROM ubuntu:21.10

LABEL maintainer='Andrew Younger'

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update --fix-missing && apt-get install -y build-essential python3 \
	python3-dev \
	python3-pip \
	&& \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/*

COPY ./scripts/setup/requirements.txt src/
RUN cd src && pip install -r requirements.txt

EXPOSE 7745

RUN rm -f /user/bin/python && ln -s /usr/bin/python3 /usr/bin/python