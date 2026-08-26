"""HTTP layer: one router per resource group.

    surveys        one survey, addressed by tenant + survey number
    tenant_surveys the same work fanned out over every survey of a tenant
    tenant_tags    tenant_tags.json
    profile        org/cx/ex tenant-profile artifacts
    catalog        process-wide reads (taxonomy, config, me, health, surveys)
    admin          the auto-retag scheduler

`run.py` owns the app, the middleware, the lifespan and the static UI, and
includes these in the order above — which is also the order the routes must be
registered in, narrowest resource first. See the "Route layout" note in run.py:
FastAPI matches in registration order, so a literal path segment has to be
registered before any `{param}` that could swallow it.
"""
