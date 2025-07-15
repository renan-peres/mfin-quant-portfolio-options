#!/bin/bash

# For each CSV file in the parent directory
cd data/
for csv_file in *.csv; do
    # Skip if no CSV files are found
    [ -e "$csv_file" ] || continue
    
    # Extract the base name without extension to use as table name
    table_name=$(basename "$csv_file" .csv)
    
    echo "Importing $csv_file into table $table_name..."
    
    # Import the CSV file into DuckDB
    duckdb data.db <<EOF
DROP TABLE IF EXISTS $table_name;
CREATE TABLE $table_name AS SELECT * FROM read_csv_auto('$csv_file', header=true);
EOF
done

echo "All CSV files from parent directory have been imported into data.db"