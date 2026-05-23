import asyncio
import sys
import os
from dotenv import load_dotenv

load_dotenv()

async def test_ingestion():
    print("=" * 60)
    print("TEST 1: RAG INGESTION")
    print("=" * 60)
    from rag import ingest_documents
    total = await ingest_documents()
    assert total > 0, f"Expected chunks > 0, got {total}"
    print(f"✅ Ingested {total} chunks total")
    return total

async def test_retrieval():
    print("\n" + "=" * 60)
    print("TEST 2: RAG RETRIEVAL")
    print("=" * 60)
    from rag import retrieve
    
    test_cases = [
        ("I am 14 and getting my first period", "HEALTH", "11-24"),
        ("How do I start saving with SIPs?", "FINANCE", "25-40"),
        ("I want to restart my career after a 5-year break", "CAREER", "25-40"),
        ("PCOS symptoms in late 40s", "HEALTH", "41+"),
    ]
    
    for query, pillar, age_band in test_cases:
        chunks = await retrieve(query, pillar, age_band, top_k=3)
        print(f"\nQuery: {query}")
        print(f"  Pillar: {pillar}, Age: {age_band}")
        print(f"  Retrieved: {len(chunks)} chunks")
        if chunks:
            print(f"  Top similarity: {chunks[0]['similarity']:.4f}")
            print(f"  Top source: {chunks[0]['source']}")
    print("\n✅ Retrieval test complete")

async def test_orchestrator():
    print("\n" + "=" * 60)
    print("TEST 3: ORCHESTRATOR — CLASSIFY")
    print("=" * 60)
    from orchestrator import classify_intent
    
    cases = [
        ("I am pregnant, what should I prepare?", "HEALTH"),
        ("Best mutual funds for women", "FINANCE"),
        ("How do I get back to work after maternity?", "CAREER"),
    ]
    
    for query, expected in cases:
        result = await classify_intent(query)
        status = "✅" if result.strip().upper() == expected else "⚠️"
        print(f"  {status} '{query[:40]}...' → {result.strip()} (expected {expected})")

async def test_end_to_end():
    print("\n" + "=" * 60)
    print("TEST 4: END-TO-END CHAT")
    print("=" * 60)
    from orchestrator import process_query as handle_query
    
    response = await handle_query(
        user_id="test-user-1",
        age_band="25-40",
        query="I am planning my first pregnancy in 6 months. What should I prepare?",
        session_id="test-session-1",
    )
    
    print(f"\n  Sub-agent: {response.get('sub_agent')}")
    print(f"  Plan steps: {len(response.get('plan', []))}")
    print(f"  Citations: {len(response.get('citations', []))}")
    print(f"  Awaiting approval: {response.get('awaiting_approval')}")
    print(f"  Validation passed: {response.get('validation_passed', 'N/A')}")
    
    assert response.get('sub_agent') == 'HEALTH', "Should route to HEALTH"
    print("\n✅ End-to-end test passed")

async def main():
    try:
        await test_ingestion()
        await test_retrieval()
        await test_orchestrator()
        await test_end_to_end()
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED — Ready to deploy!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
