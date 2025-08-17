#!/bin/bash

rm -rf build
cp -r * ~/Library/Rime/
cd ~/Library/Rime/
/Library/Input\ Methods/Squirrel.app/Contents/MacOS/Squirrel --reload
