#!/bin/bash
app="titanic.ml"
docker build -t ${app} .
docker run -d --name mytitanicmlcontainer -p 80:80 ${app}