# initialize conda environment
init:
	@echo "Setting up the virtual environment"
	conda env create -f environment.yml
	@echo 'Done!'

man_gen_env:
	@echo "Setting up the virtual environment"
	conda env create -f requirements.yml
	@echo 'Done!'

# create a directory
create_directory:
	@echo "Creating 'figs' folder, if it does not already exist."
	mkdir -p figs
	@echo "Done!"

# Generate Graphic 1
q1:
	@echo "Generating Graphic..." 	
	python -B src/q1.py
	@echo "Graphic generated!!"

# Generate Graphic 2
q2_a:
	@echo "Generating Graphic..." 	
	python -B src/q2_vis.py
	@echo "Graphic generated!!"

# Execute Sklearn model
q2_b:
	@echo "Executing Sklearn model..." 	
	python -B src/q2_sklearn.py
	@echo "Complete!!"

# Execute statsmodels model,  generate figs
q2_c:
	@echo "Executing statsmodel model...generating graphics..." 	
	python -B src/q2_stats_model.py
	@echo "Complete!!"

# Execute sklearn models, compare results
q3:
	@echo "Executing models, generating figs..." 	
	python -B src/q3.py
	@echo "Complete!!"

# Find best model a
q4:
	@echo "Executing models, generating figs..." 	
	python -B src/q4.py
	@echo "Complete!!"

# Find best model b
q5:
	@echo "Executing models, generating figs..."
	python -B src/q5.py
	@echo "Complete!!"

# Remove conda environment
clean:
	@echo "Removing env..."
	conda env remove -n assignment_env
	conda env remove -n quick_environment
	@echo "Done!!!"

.PHONY: init plot update