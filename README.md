# Temporal Tracking of Carbon Emissions in Power Storage Systems

This repository contains scripts and models for temporal tracking of carbon emissions in power storage systems. It's the underlying work for Konstantin Pelz` master theis with the following title:

> Enhancing Carbon Emission Tracing in European Electricity Markets through Temporal Carbon Tracking of Power Storage

For further questions and feedback, reach out to [Konstantin Pelz](mailto:konstantin.pelz@montelgroup.com).

## Setup

### Create virtual environment

```bash
python3 -m venv venv
```

### Activate virtual environment

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip3 install -r requirements.txt
```

### Set EQ API key

In order to use EQ's API, an API key is required. If you need one, please contact [Konstantin Pelz](mailto:konstantin.pelz@montelgroup.com).

The file must be named `eq_api_key.txt`. Just run the following command to create this file filled with your API key:

```bash
echo "your-api-key" > eq_api_key.txt
```

### Ignore Jupyter output in Git

To ignore the output of the notebooks to be comitted, run the following command

```bash
git config filter.strip-notebook-output.clean 'jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to=notebook --stdin --stdout --log-level=ERROR'
```

Read more about it [here](https://stackoverflow.com/questions/28908319/how-to-clear-jupyter-notebooks-output-in-all-cells-from-the-linux-terminal/58004619#58004619).

## Structure

The repository is structured as following:

### `/notebooks`

Jupyter notebooks for the computation of the models.

### `/src`

Python scripts to for modelling.

### `/data`

Directory to store model results. It's ignored and not in the Git repo.