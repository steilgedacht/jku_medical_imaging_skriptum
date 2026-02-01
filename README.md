# Medical Imaging Skriptum

This is a Skriptum for the Medical Imaging Lecutre on JKU in 2026. It shows all of the handwritten formulas and derivations in typst (modern fast latex). Contribution is very welcome!

## How to contribute

Simple option: Just edit the text in the files in the chapters folder. As soon as you push to the repository, a Github action automatically recompiles and pushes the Skriptum, so no need for a dedicated setup!

Dedicated Setup: 
1. Download this repo
2. Install [Typst](https://typst.app/open-source/#download-h2) or for example via the command line:
```
sudo snap install typst
```
3. Execute this so that as soon as you make changes to any of the files the pdf get automatically recompiled.
```
typst watch hello.typ 
```
4. Edit the files in the chapters directory with your favorite editor.
5. As soon as you want to commit, push and open up a pull request. If you dont want the Github action to recompile the document when uploading, your commit should contain the word "nocomp" at the start of the commit message.
