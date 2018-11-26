#!/bin/bash

git_username="rafaelsandroni"
git_password="Sandroni09!"
git_url="github.com/rafaelsandroni/models.git"
msg="reports baseline1"

git config --global user.email $git_username
git config --global user.name "Rafael"

echo $git_username

git add .
git commit -m "$msg"
git push https://$git_username:$git_password@$git_url --all




