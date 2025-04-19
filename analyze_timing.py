#!/usr/bin/env python
"""
Parse Django server logs to analyze RAG service timing data.
Run this script after your chat session to see timing statistics.

Usage:
    1. Save your Django server logs to a file: run_server.bat > logs.txt
    2. Run this script: python analyze_timing.py logs.txt
"""

import sys
import re
import statistics
from collections import defaultdict

def parse_timing_data(log_file):
    """Parse timing data from log file"""
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract request timing
    request_pattern = r'\[TIMER\] SUMMARY: Total: (\d+\.\d+)s \| RAG: (\d+\.\d+)s \| Used documents: (\w+)'
    request_matches = re.findall(request_pattern, content)
    
    # Extract RAG timing
    rag_pattern = r'\[RAG\] SUMMARY: Total: (\d+\.\d+)s \| Doc Check: (\d+\.\d+)s \| Generation: (\d+\.\d+)s \| Used docs: (\w+)'
    rag_matches = re.findall(rag_pattern, content)
    
    # Extract embedding timing
    embed_pattern = r'\[EMBED\] \[\d+:\d+:\d+\.\d+\] - Created embedding in (\d+\.\d+)s'
    embed_matches = re.findall(embed_pattern, content)
    
    # Extract Pinecone query timing
    pinecone_pattern = r'\[EMBED\] \[\d+:\d+:\d+\.\d+\] - Pinecone query completed in (\d+\.\d+)s'
    pinecone_matches = re.findall(pinecone_pattern, content)
    
    # Extract OpenAI API call timing
    openai_pattern = r'⏱️ OpenAI API call took (\d+\.\d+) seconds'
    openai_matches = re.findall(openai_pattern, content)
    
    # Extract context retrieval timing
    context_pattern = r'⏱️ Context retrieval took (\d+\.\d+) seconds'
    context_matches = re.findall(context_pattern, content)
    
    # Organize by request type
    with_docs = []
    without_docs = []
    
    for req in request_matches:
        total, rag, used_docs = float(req[0]), float(req[1]), req[2] == 'True'
        if used_docs:
            with_docs.append((total, rag))
        else:
            without_docs.append((total, rag))
    
    return {
        'requests': {
            'all': [(float(r[0]), float(r[1]), r[2] == 'True') for r in request_matches],
            'with_docs': with_docs,
            'without_docs': without_docs
        },
        'rag': [(float(r[0]), float(r[1]), float(r[2]), r[3] == 'True') for r in rag_matches],
        'embeddings': [float(e) for e in embed_matches],
        'pinecone': [float(p) for p in pinecone_matches],
        'openai': [float(o) for o in openai_matches],
        'context_retrieval': [float(c) for c in context_matches]
    }

def calc_stats(data_list):
    """Calculate statistics for a list of timing data"""
    if not data_list:
        return {'min': 0, 'max': 0, 'avg': 0, 'median': 0, 'count': 0}
    
    return {
        'min': min(data_list),
        'max': max(data_list),
        'avg': sum(data_list) / len(data_list),
        'median': statistics.median(data_list) if len(data_list) > 1 else data_list[0],
        'count': len(data_list)
    }

def analyze_timing(timing_data):
    """Analyze timing data and generate statistics"""
    results = {}
    
    # Overall request timing
    all_requests = timing_data['requests']['all']
    if all_requests:
        results['overall_request'] = {
            'total': calc_stats([r[0] for r in all_requests]),
            'rag': calc_stats([r[1] for r in all_requests]),
            'count': len(all_requests),
            'with_docs_count': len([r for r in all_requests if r[2]]),
            'without_docs_count': len([r for r in all_requests if not r[2]])
        }
    
    # Requests with documents
    with_docs = timing_data['requests']['with_docs']
    if with_docs:
        results['requests_with_docs'] = {
            'total': calc_stats([r[0] for r in with_docs]),
            'rag': calc_stats([r[1] for r in with_docs])
        }
    
    # Requests without documents
    without_docs = timing_data['requests']['without_docs']
    if without_docs:
        results['requests_without_docs'] = {
            'total': calc_stats([r[0] for r in without_docs]),
            'rag': calc_stats([r[1] for r in without_docs])
        }
    
    # RAG service timing
    rag = timing_data['rag']
    if rag:
        results['rag_service'] = {
            'total': calc_stats([r[0] for r in rag]),
            'doc_check': calc_stats([r[1] for r in rag]),
            'generation': calc_stats([r[2] for r in rag])
        }
    
    # Component timing
    for component in ['embeddings', 'pinecone', 'openai', 'context_retrieval']:
        if timing_data[component]:
            results[component] = calc_stats(timing_data[component])
    
    return results

def format_time(seconds):
    """Format time in seconds to a readable string"""
    if seconds < 0.1:
        return f"{seconds*1000:.1f}ms"
    return f"{seconds:.2f}s"

def print_stats(stats, label):
    """Print statistics in a readable format"""
    print(f"\n{label} (count: {stats['count']}):")
    print(f"  Min: {format_time(stats['min'])}")
    print(f"  Max: {format_time(stats['max'])}")
    print(f"  Avg: {format_time(stats['avg'])}")
    print(f"  Median: {format_time(stats['median'])}")

def print_results(results):
    """Print analysis results in a readable format"""
    print("\n========== RAG TIMING ANALYSIS ==========")
    
    if 'overall_request' in results:
        overall = results['overall_request']
        print(f"\nOVERALL REQUESTS (total: {overall['count']}):")
        print(f"  With documents: {overall['with_docs_count']}")
        print(f"  Without documents: {overall['without_docs_count']}")
        print_stats(overall['total'], "Total request time")
        print_stats(overall['rag'], "RAG service time")
    
    if 'requests_with_docs' in results:
        print("\nREQUESTS WITH DOCUMENTS:")
        print_stats(results['requests_with_docs']['total'], "Total request time")
        print_stats(results['requests_with_docs']['rag'], "RAG service time")
    
    if 'requests_without_docs' in results:
        print("\nREQUESTS WITHOUT DOCUMENTS:")
        print_stats(results['requests_without_docs']['total'], "Total request time")
        print_stats(results['requests_without_docs']['rag'], "RAG service time")
    
    if 'rag_service' in results:
        rag = results['rag_service']
        print("\nRAG SERVICE COMPONENTS:")
        print_stats(rag['total'], "Total RAG time")
        print_stats(rag['doc_check'], "Document check time")
        print_stats(rag['generation'], "Response generation time")
    
    for component, stats in results.items():
        if component not in ['overall_request', 'requests_with_docs', 'requests_without_docs', 'rag_service']:
            print_stats(stats, f"{component.upper()} time")
    
    print("\n============== SUMMARY ===============")
    if 'openai' in results and 'overall_request' in results:
        openai_pct = results['openai']['avg'] / results['overall_request']['total']['avg'] * 100
        print(f"OpenAI API calls account for approximately {openai_pct:.1f}% of total request time")
    
    if 'embeddings' in results and 'overall_request' in results:
        embed_pct = results['embeddings']['avg'] / results['overall_request']['total']['avg'] * 100
        print(f"Embedding generation accounts for approximately {embed_pct:.1f}% of total request time")
    
    if 'pinecone' in results and 'overall_request' in results:
        pinecone_pct = results['pinecone']['avg'] / results['overall_request']['total']['avg'] * 100
        print(f"Pinecone queries account for approximately {pinecone_pct:.1f}% of total request time")
    
    print("\n=======================================")

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_timing.py <log_file>")
        return
    
    log_file = sys.argv[1]
    try:
        timing_data = parse_timing_data(log_file)
        results = analyze_timing(timing_data)
        print_results(results)
    except Exception as e:
        print(f"Error analyzing log file: {str(e)}")

if __name__ == "__main__":
    main() 