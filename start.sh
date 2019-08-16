#!/usr/bin/env bash

./docker-clean.sh
./build-task-images.sh 0.1
docker-compose up orchestrator
