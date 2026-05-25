import rdflib
from pyshacl import validate
from rdflib.namespace import RDF, RDFS, XSD

# ================= 配置区域 =================
TTL_FILES = [
    "data/test_intermediate_results_2/SS_03/SS_03_KnowledgeGraph.ttl",
]

# 你的 Ontology Namespace
TARGET_NAMESPACE = "http://www.semanticweb.org/yanha/ontologies/2025/7/untitled-ontology-26/"

# ================= SHACL 规则定义 =================

# 规则 1: Dimension 完整性 (必须有 Unit，且有 Decimal 或 String 值)
SHACL_RULE_1 = f"""
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix ex: <{TARGET_NAMESPACE}> .

ex:DimensionIntegrityShape
    a sh:NodeShape ;
    sh:targetClass ex:Dimension ;
    sh:property [
        sh:path ex:hasDimensionUnit ;
        sh:minCount 1 ;
        sh:message "Missing Unit" ;
    ] ;
    sh:or (
        [ sh:property [ sh:path ex:hasDimensionDecimalValue ; sh:minCount 1 ; ] ]
        [ sh:property [ sh:path ex:hasDimensionStringValue ; sh:minCount 1 ; ] ]
    ) .
"""

# 规则 2: 设备标识完整性 (必须有 Model Name)
# 注意：这里 TargetClass 设为 DeviceComponent，请确保你的实例是这个类的实例
SHACL_RULE_2 = f"""
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <{TARGET_NAMESPACE}> .

ex:DeviceIdentityShape
    a sh:NodeShape ;
    sh:targetClass ex:DeviceComponent ;
    sh:property [
        sh:path ex:hasModelName ;
        sh:minCount 1 ;
        sh:message "Device Missing Model Name" ;
    ] .
"""

# 规则 3: 连通性检查 (孤立节点检测)
# 检查 Dimension 是否被某个设备通过 hasDimension 引用 (使用反向路径检查)
SHACL_RULE_3 = f"""
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <{TARGET_NAMESPACE}> .

ex:ConnectivityShape
    a sh:NodeShape ;
    sh:targetClass ex:Dimension ;
    sh:property [
        sh:path [ sh:inversePath ex:hasDimension ] ;
        sh:minCount 1 ;
        sh:message "Orphan Dimension (Not linked to any Device)" ;
    ] .
"""

# 定义评估任务列表: (任务名称, 目标类名, SHACL规则字符串)
VALIDATION_TASKS = [
    ("Rule 1: Dim Integrity", "Dimension", SHACL_RULE_1),
    ("Rule 2: Device Identity", "DeviceComponent", SHACL_RULE_2),
    ("Rule 3: Connectivity", "Dimension", SHACL_RULE_3),
]

def run_task(data_graph, task_name, target_class_name, shacl_text):
    """运行单个 SHACL 任务并返回统计数据"""
    
    # 1. 统计分母 (Total Instances)
    query_count = f"""
    PREFIX ex: <{TARGET_NAMESPACE}>
    SELECT (COUNT(?s) AS ?qty)
    WHERE {{
        ?s a ex:{target_class_name} .
    }}
    """
    try:
        res = data_graph.query(query_count)
        total = 0
        for row in res:
            total = int(row.qty)
    except Exception as e:
        print(f"    [Error querying count]: {e}")
        return

    # 如果没有实例，直接返回
    if total == 0:
        print(f"    {task_name}: ⚠️ No instances of {target_class_name} found.")
        return

    # 2. 运行 SHACL
    shacl_graph = rdflib.Graph()
    shacl_graph.parse(data=shacl_text, format="turtle")
    
    conforms, report_graph, _ = validate(
        data_graph,
        shacl_graph=shacl_graph,
        inference='rdfs',
        serialize_report_graph=False
    )

    # 3. 统计分子 (Violations)
    query_viol = """
    PREFIX sh: <http://www.w3.org/ns/shacl#>
    SELECT (COUNT(?s) AS ?qty) WHERE { ?s a sh:ValidationResult . }
    """
    res_v = report_graph.query(query_viol)
    violations = 0
    for row in res_v:
        violations = int(row.qty)

    # 4. 计算并打印
    rate = (violations / total) * 100
    status_icon = "✅" if violations == 0 else "❌"
    
    print(f"    {task_name:<25} | Total: {total:<4} | Violations: {violations:<3} | Rate: {rate:.2f}% {status_icon}")

def analyze_file(file_path):
    print(f"\n{'='*60}")
    print(f"📄 Analyzing File: {file_path}")
    print(f"{'='*60}")
    
    # 加载数据文件
    data_graph = rdflib.Graph()
    try:
        data_graph.parse(file_path, format="turtle")
    except Exception as e:
        print(f"❌ Error loading TTL: {e}")
        return

    # 依次运行三个规则
    for name, target_class, rule_text in VALIDATION_TASKS:
        run_task(data_graph, name, target_class, rule_text)

if __name__ == "__main__":
    print(f"Target Namespace: {TARGET_NAMESPACE}")
    for f in TTL_FILES:
        analyze_file(f)