# ContextChain v2.0 API Documentation

## Overview

ContextChain v2.0 provides a comprehensive RESTful API for intelligent context optimization in LLM applications. The API enables adaptive budget allocation, multi-vector retrieval, and RL-based compression through simple HTTP endpoints.

**Base URL**: `https://api.contextchain.ai/v2`  
**Authentication**: Bearer token (API Key)  
**Content-Type**: `application/json`  
**Rate Limits**: 1000 requests/minute (configurable per plan)

---

## Authentication

Include your API key in the Authorization header:

```bash
curl -H "Authorization: Bearer sk-cc-abc123..." \
     -H "Content-Type: application/json" \
     https://api.contextchain.ai/v2/query