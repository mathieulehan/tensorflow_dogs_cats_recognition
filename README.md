# Cats & dogs recognition - Tensorflow

Images are obviously not included within this repository.
Check this : https://www.kaggle.com/c/dogs-vs-cats/data

You will have to include the unzipped /train folder in the project's root.

## Dataset configuration

- Launch preProcessIntoStandardDirectories to create a test folder.
- Launch createFinalDogs&CatsDatasets to separate cats and dogs images into differents folders.

## Usage

Once you have all necessay images and folders, you can :
- generate and save a one block model (with convolutions) using baseCnnModel.py
- generate and save a three blocks model (with convolutions) using threeBlocksVggModel.py
- generate and save a model with dropout regularization using preProcessIntoStandardDirectories.py

## License
[MIT](https://choosealicense.com/licenses/mit/)