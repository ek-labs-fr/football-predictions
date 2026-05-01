const DATA_BUCKET = 'fpingeststack-databuckete3889a50-1rbnetcmk2gj';
const REGION = 'eu-west-3';

export async function onRequest(context) {
  const { request, params } = context;
  const path = Array.isArray(params.path) ? params.path.join('/') : params.path;
  const url = `https://${DATA_BUCKET}.s3.${REGION}.amazonaws.com/web/data/${path}`;

  const upstream = await fetch(url, {
    method: 'GET',
    headers: { Accept: request.headers.get('Accept') ?? '*/*' },
  });

  if (!upstream.ok) {
    return new Response(`Data fetch failed: ${upstream.status}`, { status: upstream.status });
  }

  return new Response(upstream.body, {
    status: 200,
    headers: {
      'Content-Type': upstream.headers.get('Content-Type') ?? 'application/json',
      'Cache-Control': 'public, max-age=300',
    },
  });
}
