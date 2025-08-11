#!/usr/bin/env python3
"""
SAS to Snowflake Migration Multi-Agent System
A comprehensive AI-driven system for migrating legacy SAS code to Snowflake
"""

import os
import json
import re
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import sqlite3
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MigrationComponent:
    """Represents a component to be migrated"""
    name: str
    type: str  # 'data_step', 'proc_sql', 'macro', 'proc'
    source_code: str
    dependencies: List[str]
    complexity_score: int
    business_rules: List[str]

@dataclass
class SchemaMapping:
    """Maps SAS dataset structure to Snowflake table"""
    sas_dataset: str
    snowflake_table: str
    column_mappings: Dict[str, Dict]  # {sas_col: {sf_col, sf_type, constraints}}
    ddl_script: str

class DiscoveryAnalysisAgent:
    """Agent responsible for discovering and analyzing SAS code components"""
    
    def __init__(self):
        self.components = []
        self.dependencies = {}
    
    def analyze_sas_program(self, sas_code: str, program_name: str) -> List[MigrationComponent]:
        """Parse SAS code and identify components"""
        logger.info(f"Analyzing SAS program: {program_name}")
        
        components = []
        
        # Split code into logical blocks
        blocks = self._split_sas_blocks(sas_code)
        
        for i, block in enumerate(blocks):
            component_type = self._identify_block_type(block)
            dependencies = self._extract_dependencies(block)
            complexity = self._calculate_complexity(block)
            business_rules = self._extract_business_rules(block)
            
            component = MigrationComponent(
                name=f"{program_name}_block_{i+1}",
                type=component_type,
                source_code=block,
                dependencies=dependencies,
                complexity_score=complexity,
                business_rules=business_rules
            )
            components.append(component)
        
        self.components.extend(components)
        return components
    
    def _split_sas_blocks(self, code: str) -> List[str]:
        """Split SAS code into logical processing blocks"""
        # Simplified: split by DATA/PROC statements
        blocks = []
        current_block = ""
        
        lines = code.split('\n')
        in_block = False
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('*'):  # Skip comments and empty lines
                continue
                
            if re.match(r'^(DATA|PROC)\s+', line, re.IGNORECASE):
                if current_block:
                    blocks.append(current_block)
                current_block = line + '\n'
                in_block = True
            elif line.upper() in ['RUN;', 'QUIT;'] and in_block:
                current_block += line + '\n'
                blocks.append(current_block)
                current_block = ""
                in_block = False
            elif in_block:
                current_block += line + '\n'
        
        if current_block:
            blocks.append(current_block)
            
        return [block for block in blocks if block.strip()]
    
    def _identify_block_type(self, block: str) -> str:
        """Identify the type of SAS block"""
        block_upper = block.upper()
        if block_upper.startswith('DATA '):
            return 'data_step'
        elif block_upper.startswith('PROC SQL'):
            return 'proc_sql'
        elif 'MACRO' in block_upper:
            return 'macro'
        elif block_upper.startswith('PROC '):
            return 'proc'
        return 'unknown'
    
    def _extract_dependencies(self, block: str) -> List[str]:
        """Extract dataset and library dependencies"""
        dependencies = []
        
        # Find dataset references (simplified pattern)
        dataset_patterns = [
            r'FROM\s+([A-Za-z_][A-Za-z0-9_.]*)',
            r'SET\s+([A-Za-z_][A-Za-z0-9_.]*)',
            r'MERGE\s+([A-Za-z_][A-Za-z0-9_.]*)'
        ]
        
        for pattern in dataset_patterns:
            matches = re.findall(pattern, block, re.IGNORECASE)
            dependencies.extend(matches)
        
        return list(set(dependencies))  # Remove duplicates
    
    def _calculate_complexity(self, block: str) -> int:
        """Calculate complexity score based on various factors"""
        complexity = 1  # Base complexity
        
        # Add complexity for various constructs
        complexity_factors = {
            r'IF\s+.*THEN': 2,  # Conditional logic
            r'DO\s+.*END;': 3,  # Loops
            r'PROC\s+SQL': 2,   # SQL procedures
            r'MACRO\s+': 4,     # Macros
            r'ARRAY\s+': 3,     # Arrays
            r'RETAIN\s+': 2,    # Retain statements
        }
        
        for pattern, score in complexity_factors.items():
            matches = len(re.findall(pattern, block, re.IGNORECASE))
            complexity += matches * score
        
        return min(complexity, 10)  # Cap at 10
    
    def _extract_business_rules(self, block: str) -> List[str]:
        """Extract business rules from comments and logic"""
        rules = []
        
        # Extract from comments
        comment_matches = re.findall(r'\*\s*([^;]+);', block)
        rules.extend([match.strip() for match in comment_matches if len(match.strip()) > 10])
        
        # Extract from conditional logic (simplified)
        if_matches = re.findall(r'IF\s+([^;]+);', block, re.IGNORECASE)
        rules.extend([f"Business rule: {match.strip()}" for match in if_matches[:3]])
        
        return rules

class SchemaTranslationAgent:
    """Agent responsible for converting SAS schemas to Snowflake"""
    
    def __init__(self):
        self.type_mappings = {
            'char': 'VARCHAR',
            'varchar': 'VARCHAR',
            'num': 'NUMBER',
            'numeric': 'NUMBER',
            'date': 'DATE',
            'datetime': 'TIMESTAMP',
            'time': 'TIME'
        }
    
    def translate_schema(self, sas_dataset_info: Dict) -> SchemaMapping:
        """Translate SAS dataset structure to Snowflake table"""
        logger.info(f"Translating schema for: {sas_dataset_info['name']}")
        
        sas_name = sas_dataset_info['name']
        snowflake_name = self._convert_table_name(sas_name)
        
        column_mappings = {}
        ddl_parts = [f"CREATE OR REPLACE TABLE {snowflake_name} ("]
        
        for col_info in sas_dataset_info.get('columns', []):
            sas_col = col_info['name']
            sas_type = col_info['type'].lower()
            sas_length = col_info.get('length', '')
            
            sf_col = self._convert_column_name(sas_col)
            sf_type = self._map_data_type(sas_type, sas_length)
            
            column_mappings[sas_col] = {
                'snowflake_column': sf_col,
                'snowflake_type': sf_type,
                'constraints': []
            }
            
            ddl_parts.append(f"    {sf_col} {sf_type},")
        
        # Remove last comma and close DDL
        if ddl_parts[-1].endswith(','):
            ddl_parts[-1] = ddl_parts[-1][:-1]
        ddl_parts.append(");")
        
        ddl_script = '\n'.join(ddl_parts)
        
        return SchemaMapping(
            sas_dataset=sas_name,
            snowflake_table=snowflake_name,
            column_mappings=column_mappings,
            ddl_script=ddl_script
        )
    
    def _convert_table_name(self, sas_name: str) -> str:
        """Convert SAS dataset name to Snowflake table name"""
        # Replace dots with underscores and make uppercase (Snowflake convention)
        return sas_name.replace('.', '_').upper()
    
    def _convert_column_name(self, sas_col: str) -> str:
        """Convert SAS column name to Snowflake column name"""
        return sas_col.upper()
    
    def _map_data_type(self, sas_type: str, length: str = '') -> str:
        """Map SAS data type to Snowflake data type"""
        base_type = self.type_mappings.get(sas_type, 'VARCHAR')
        
        if base_type == 'VARCHAR' and length:
            return f"VARCHAR({length})"
        elif base_type == 'NUMBER' and length:
            return f"NUMBER({length})"
        
        return base_type

class CodeConversionAgent:
    """Agent responsible for converting SAS code to Snowflake SQL"""
    
    def convert_data_step(self, component: MigrationComponent, schema_mapping: SchemaMapping) -> str:
        """Convert SAS DATA step to Snowflake SQL"""
        logger.info(f"Converting DATA step: {component.name}")
        
        sas_code = component.source_code
        
        # Extract basic structure
        data_match = re.search(r'DATA\s+([^;]+);', sas_code, re.IGNORECASE)
        set_match = re.search(r'SET\s+([^;]+);', sas_code, re.IGNORECASE)
        
        if not data_match:
            return "-- Could not parse DATA statement"
        
        output_table = data_match.group(1).strip()
        input_table = set_match.group(1).strip() if set_match else "source_table"
        
        # Convert to CREATE TABLE AS SELECT
        snowflake_sql = f"""
-- Converted from SAS DATA step: {component.name}
CREATE OR REPLACE TABLE {self._convert_table_name(output_table)} AS
SELECT 
    *,
    -- Add any calculated fields here
    CURRENT_TIMESTAMP() as CREATED_TIMESTAMP
FROM {self._convert_table_name(input_table)}
"""
        
        # Add WHERE clause if present
        where_match = re.search(r'WHERE\s+([^;]+);', sas_code, re.IGNORECASE)
        if where_match:
            condition = where_match.group(1).strip()
            snowflake_sql += f"WHERE {self._convert_condition(condition)}\n"
        
        snowflake_sql += ";"
        
        return snowflake_sql
    
    def convert_proc_sql(self, component: MigrationComponent) -> str:
        """Convert SAS PROC SQL to Snowflake SQL"""
        logger.info(f"Converting PROC SQL: {component.name}")
        
        sas_code = component.source_code
        
        # Extract SQL statements between PROC SQL and QUIT
        sql_pattern = r'PROC\s+SQL;(.*?)QUIT;'
        match = re.search(sql_pattern, sas_code, re.DOTALL | re.IGNORECASE)
        
        if not match:
            return "-- Could not extract SQL from PROC SQL block"
        
        sql_content = match.group(1).strip()
        
        # Basic conversions for Snowflake
        converted_sql = self._apply_snowflake_conversions(sql_content)
        
        return f"""
-- Converted from SAS PROC SQL: {component.name}
{converted_sql}
"""
    
    def _convert_table_name(self, sas_name: str) -> str:
        """Convert SAS table name to Snowflake format"""
        return sas_name.replace('.', '_').upper()
    
    def _convert_condition(self, condition: str) -> str:
        """Convert SAS WHERE condition to Snowflake SQL"""
        # Basic conversions (extend as needed)
        converted = condition
        converted = re.sub(r'\bEQ\b', '=', converted, flags=re.IGNORECASE)
        converted = re.sub(r'\bNE\b', '<>', converted, flags=re.IGNORECASE)
        converted = re.sub(r'\bGT\b', '>', converted, flags=re.IGNORECASE)
        converted = re.sub(r'\bLT\b', '<', converted, flags=re.IGNORECASE)
        return converted
    
    def _apply_snowflake_conversions(self, sql: str) -> str:
        """Apply Snowflake-specific SQL conversions"""
        converted = sql
        
        # Convert SAS-specific functions to Snowflake equivalents
        function_mappings = {
            r'\bTODAY\(\)': 'CURRENT_DATE()',
            r'\bDATETIME\(\)': 'CURRENT_TIMESTAMP()',
            r'\bPUT\(': 'TO_VARCHAR(',
            r'\bINPUT\(': 'TO_NUMBER(',
        }
        
        for sas_func, sf_func in function_mappings.items():
            converted = re.sub(sas_func, sf_func, converted, flags=re.IGNORECASE)
        
        return converted

class BusinessLogicPreservationAgent:
    """Agent responsible for ensuring business logic is preserved"""
    
    def validate_conversion(self, original_component: MigrationComponent, 
                          converted_sql: str) -> Dict[str, any]:
        """Validate that business logic is preserved in conversion"""
        logger.info(f"Validating business logic for: {original_component.name}")
        
        validation_results = {
            'component_name': original_component.name,
            'business_rules_preserved': [],
            'potential_issues': [],
            'confidence_score': 0.8,  # Default confidence
            'recommendations': []
        }
        
        # Check if business rules are reflected in converted code
        for rule in original_component.business_rules:
            if self._check_rule_preservation(rule, converted_sql):
                validation_results['business_rules_preserved'].append(rule)
            else:
                validation_results['potential_issues'].append(f"Rule may not be preserved: {rule}")
        
        # Add recommendations based on complexity
        if original_component.complexity_score > 7:
            validation_results['recommendations'].append("High complexity - recommend manual review")
            validation_results['confidence_score'] -= 0.2
        
        return validation_results
    
    def _check_rule_preservation(self, rule: str, converted_sql: str) -> bool:
        """Check if a business rule is preserved in converted SQL"""
        # Simplified check - look for key terms from the rule in the SQL
        rule_words = re.findall(r'\w+', rule.lower())
        sql_lower = converted_sql.lower()
        
        preserved_words = sum(1 for word in rule_words if word in sql_lower)
        return preserved_words > len(rule_words) * 0.6  # 60% threshold

class TestingValidationAgent:
    """Agent responsible for generating tests and validation"""
    
    def generate_test_cases(self, component: MigrationComponent, 
                          converted_sql: str) -> List[Dict]:
        """Generate test cases for validating the conversion"""
        logger.info(f"Generating test cases for: {component.name}")
        
        test_cases = []
        
        # Generate data validation test
        test_cases.append({
            'test_type': 'data_validation',
            'description': f'Validate data consistency for {component.name}',
            'sas_query': self._generate_sas_validation_query(component),
            'snowflake_query': self._generate_snowflake_validation_query(converted_sql),
            'expected_result': 'Row counts and key metrics should match'
        })
        
        # Generate business rule test
        for rule in component.business_rules[:2]:  # Limit to first 2 rules
            test_cases.append({
                'test_type': 'business_rule',
                'description': f'Validate business rule: {rule[:50]}...',
                'test_query': self._generate_rule_test_query(rule, converted_sql),
                'expected_result': 'Business rule logic should be maintained'
            })
        
        return test_cases
    
    def _generate_sas_validation_query(self, component: MigrationComponent) -> str:
        """Generate SAS query for validation"""
        return f"""
PROC SQL;
SELECT COUNT(*) as row_count,
       COUNT(DISTINCT *) as distinct_rows,
       MIN(created_date) as min_date,
       MAX(created_date) as max_date
FROM {component.name};
QUIT;
"""
    
    def _generate_snowflake_validation_query(self, converted_sql: str) -> str:
        """Generate Snowflake query for validation"""
        return """
SELECT COUNT(*) as row_count,
       COUNT(DISTINCT *) as distinct_rows,
       MIN(created_date) as min_date,
       MAX(created_date) as max_date
FROM converted_table;
"""
    
    def _generate_rule_test_query(self, rule: str, converted_sql: str) -> str:
        """Generate test query for business rule validation"""
        return f"""
-- Test for business rule: {rule}
SELECT 
    CASE 
        WHEN COUNT(*) > 0 THEN 'PASS'
        ELSE 'FAIL'
    END as test_result
FROM ({converted_sql.rstrip(';')}) 
WHERE /* Add specific business rule validation logic here */;
"""

class OrchestrationAgent:
    """Main orchestration agent that coordinates the migration process"""
    
    def __init__(self):
        self.discovery_agent = DiscoveryAnalysisAgent()
        self.schema_agent = SchemaTranslationAgent()
        self.code_agent = CodeConversionAgent()
        self.business_agent = BusinessLogicPreservationAgent()
        self.testing_agent = TestingValidationAgent()
        
        self.migration_results = {
            'components': [],
            'schemas': [],
            'conversions': [],
            'validations': [],
            'test_cases': [],
            'summary': {}
        }
    
    def execute_migration(self, sas_programs: Dict[str, str], 
                         dataset_schemas: List[Dict]) -> Dict:
        """Execute the complete migration process"""
        logger.info("Starting SAS to Snowflake migration process")
        
        # Phase 1: Discovery and Analysis
        logger.info("Phase 1: Discovery and Analysis")
        all_components = []
        for program_name, sas_code in sas_programs.items():
            components = self.discovery_agent.analyze_sas_program(sas_code, program_name)
            all_components.extend(components)
        
        self.migration_results['components'] = [
            {
                'name': comp.name,
                'type': comp.type,
                'complexity': comp.complexity_score,
                'dependencies': comp.dependencies,
                'business_rules_count': len(comp.business_rules)
            }
            for comp in all_components
        ]
        
        # Phase 2: Schema Translation
        logger.info("Phase 2: Schema Translation")
        schema_mappings = []
        for schema_info in dataset_schemas:
            mapping = self.schema_agent.translate_schema(schema_info)
            schema_mappings.append(mapping)
        
        self.migration_results['schemas'] = [
            {
                'sas_dataset': mapping.sas_dataset,
                'snowflake_table': mapping.snowflake_table,
                'ddl_script': mapping.ddl_script
            }
            for mapping in schema_mappings
        ]
        
        # Phase 3: Code Conversion
        logger.info("Phase 3: Code Conversion")
        conversions = []
        for component in all_components:
            schema_mapping = schema_mappings[0] if schema_mappings else None
            
            if component.type == 'data_step':
                converted_sql = self.code_agent.convert_data_step(component, schema_mapping)
            elif component.type == 'proc_sql':
                converted_sql = self.code_agent.convert_proc_sql(component)
            else:
                converted_sql = f"-- {component.type} conversion not implemented yet"
            
            conversions.append({
                'component_name': component.name,
                'original_code': component.source_code,
                'converted_sql': converted_sql
            })
        
        self.migration_results['conversions'] = conversions
        
        # Phase 4: Business Logic Validation
        logger.info("Phase 4: Business Logic Validation")
        validations = []
        for i, component in enumerate(all_components):
            if i < len(conversions):
                validation = self.business_agent.validate_conversion(
                    component, conversions[i]['converted_sql']
                )
                validations.append(validation)
        
        self.migration_results['validations'] = validations
        
        # Phase 5: Test Case Generation
        logger.info("Phase 5: Test Case Generation")
        all_test_cases = []
        for i, component in enumerate(all_components):
            if i < len(conversions):
                test_cases = self.testing_agent.generate_test_cases(
                    component, conversions[i]['converted_sql']
                )
                all_test_cases.extend(test_cases)
        
        self.migration_results['test_cases'] = all_test_cases
        
        # Generate Summary
        self.migration_results['summary'] = {
            'total_components': len(all_components),
            'total_schemas': len(schema_mappings),
            'high_complexity_components': len([c for c in all_components if c.complexity_score > 7]),
            'average_confidence': sum(v['confidence_score'] for v in validations) / len(validations) if validations else 0,
            'total_test_cases': len(all_test_cases),
            'migration_timestamp': datetime.now().isoformat()
        }
        
        logger.info("Migration process completed successfully")
        return self.migration_results
    
    def generate_migration_report(self) -> str:
        """Generate a comprehensive migration report"""
        results = self.migration_results
        
        report = f"""
# SAS to Snowflake Migration Report
Generated: {results['summary']['migration_timestamp']}

## Migration Summary
- Total Components Analyzed: {results['summary']['total_components']}
- Schema Translations: {results['summary']['total_schemas']}
- High Complexity Components: {results['summary']['high_complexity_components']}
- Average Confidence Score: {results['summary']['average_confidence']:.2f}
- Test Cases Generated: {results['summary']['total_test_cases']}

## Component Analysis
"""
        
        for comp in results['components']:
            report += f"""
### {comp['name']} ({comp['type']})
- Complexity Score: {comp['complexity']}/10
- Dependencies: {', '.join(comp['dependencies']) if comp['dependencies'] else 'None'}
- Business Rules: {comp['business_rules_count']}
"""
        
        report += "\n## Schema Translations\n"
        for schema in results['schemas']:
            report += f"""
### {schema['sas_dataset']} → {schema['snowflake_table']}
```sql
{schema['ddl_script']}
```
"""
        
        report += "\n## Validation Results\n"
        for validation in results['validations']:
            report += f"""
### {validation['component_name']}
- Confidence Score: {validation['confidence_score']:.2f}
- Business Rules Preserved: {len(validation['business_rules_preserved'])}
- Potential Issues: {len(validation['potential_issues'])}
"""
            
            if validation['recommendations']:
                report += f"- Recommendations: {'; '.join(validation['recommendations'])}\n"
        
        return report

def create_sample_project():
    """Create sample SAS programs and datasets for demonstration"""
    
    # Sample SAS programs
    sample_programs = {
        "customer_analysis": """
        * Customer data analysis program;
        * Business rule: Only include active customers with revenue > 1000;
        DATA work.active_customers;
            SET lib.customer_master;
            WHERE status = 'ACTIVE' AND annual_revenue > 1000;
            
            * Calculate customer score;
            IF annual_revenue > 10000 THEN customer_tier = 'PREMIUM';
            ELSE IF annual_revenue > 5000 THEN customer_tier = 'GOLD';
            ELSE customer_tier = 'STANDARD';
            
            * Calculate days since last order;
            days_since_order = TODAY() - last_order_date;
        RUN;
        
        PROC SQL;
            CREATE TABLE work.customer_summary AS
            SELECT customer_tier,
                   COUNT(*) as customer_count,
                   AVG(annual_revenue) as avg_revenue,
                   MAX(days_since_order) as max_days_since_order
            FROM work.active_customers
            GROUP BY customer_tier
            ORDER BY avg_revenue DESC;
        QUIT;
        """,
        
        "sales_reporting": """
        * Sales reporting and aggregation;
        * Business rule: Exclude returns and cancelled orders;
        DATA work.valid_sales;
            SET lib.sales_transactions;
            WHERE order_status NOT IN ('CANCELLED', 'RETURNED');
            
            * Calculate net amount;
            net_amount = gross_amount - discount_amount - tax_amount;
            
            * Categorize order size;
            IF net_amount > 1000 THEN order_size = 'LARGE';
            ELSE IF net_amount > 500 THEN order_size = 'MEDIUM';
            ELSE order_size = 'SMALL';
        RUN;
        
        PROC SQL;
            CREATE TABLE work.monthly_sales AS
            SELECT YEAR(order_date) as sales_year,
                   MONTH(order_date) as sales_month,
                   order_size,
                   SUM(net_amount) as total_sales,
                   COUNT(*) as order_count
            FROM work.valid_sales
            GROUP BY sales_year, sales_month, order_size;
        QUIT;
        """
    }
    
    # Sample dataset schemas
    sample_schemas = [
        {
            'name': 'lib.customer_master',
            'columns': [
                {'name': 'customer_id', 'type': 'num', 'length': '8'},
                {'name': 'customer_name', 'type': 'char', 'length': '100'},
                {'name': 'status', 'type': 'char', 'length': '10'},
                {'name': 'annual_revenue', 'type': 'num', 'length': '12.2'},
                {'name': 'last_order_date', 'type': 'date'},
                {'name': 'created_date', 'type': 'datetime'}
            ]
        },
        {
            'name': 'lib.sales_transactions',
            'columns': [
                {'name': 'transaction_id', 'type': 'num', 'length': '8'},
                {'name': 'customer_id', 'type': 'num', 'length': '8'},
                {'name': 'order_date', 'type': 'date'},
                {'name': 'order_status', 'type': 'char', 'length': '20'},
                {'name': 'gross_amount', 'type': 'num', 'length': '12.2'},
                {'name': 'discount_amount', 'type': 'num', 'length': '12.2'},
                {'name': 'tax_amount', 'type': 'num', 'length': '12.2'}
            ]
        }
    ]
    
    return sample_programs, sample_schemas

def main():
    """Main function to demonstrate the migration system"""
    print("=== SAS to Snowflake Migration Multi-Agent System ===\n")
    
    # Create sample project data
    sample_programs, sample_schemas = create_sample_project()
    
    # Initialize orchestration agent
    orchestrator = OrchestrationAgent()
    
    # Execute migration
    print("Starting migration process...\n")
    migration_results = orchestrator.execute_migration(sample_programs, sample_schemas)
    
    # Generate and display report
    print("Generating migration report...\n")
    report = orchestrator.generate_migration_report()
    print(report)
    
    # Save results to JSON file
    output_file = f"migration_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(migration_results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: {output_file}")
    print("\n=== Migration Complete ===")

if __name__ == "__main__":
    main()
