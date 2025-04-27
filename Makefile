help:
	@echo "Run 'make regression' to run regression tests"

# if COVERAGE is defined, use it to run the first test
# and also to print a report at the end
#
# Unittests rewrite the coverage data and the regtests makefile
# runs them so they append to the coverage data
regression:
	$(MAKE) -f tests/regressiontests/Makefile check
	@echo '* * * * * * * * * * * * * * * * * * * * *'
	@echo ' Rejoice, tests have passed successfully!'
	@echo '* * * * * * * * * * * * * * * * * * * * *'
	$(MAKE) -f tests/regressiontests/Makefile clean

