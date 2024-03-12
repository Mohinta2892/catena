import json
import numpy as np
import urllib
import zlib
import logging

logger = logging.getLogger(__name__)


def print_compression_statistics(compression_name, pre, post):
    pre_size = len(pre)
    post_size = len(post)
    print('{type}: {before}->{after}, saving {total} bytes ({percent:.2f}%).'
        .format(
        type=compression_name,
        before=pre_size,
        after=post_size,
        total=post_size - pre_size,
        percent=(100 - pre_size * 100. / post_size),
    ))


def query_seg_ids(locations, token, volume_id):
    # Set up request headers.
    headers = {
        'Authorization': 'Bearer ' + token,
        'Content-type': 'application/json',
        'Accept-Encoding': 'gzip',
    }

    # Since JavaScript can't handle uint64 datatypes (I know, right?), we have to
    # encode them as strings :(
    locations_as_string = [','.join(x) for x in np.char.mod('%u', locations)]

    logger.debug(('Requesting values for volume {volume_id}').format(
        volume_id=volume_id))

    # Details of the request. Note that due to a JSON standard limitation, the
    # corner and size must be represented as an X,Y,Z string tuple.
    request = {'locations': locations_as_string}
    logger.debug('Request body: ' + json.dumps(request))

    # Now make the HTTP call.
    url = 'https://brainmaps.googleapis.com/v1/volumes/{volume_id}/values'.format(
        volume_id=volume_id)
    req = urllib.request.Request(url, json.dumps(request).encode('utf8'),
                                 headers)
    compressed_data = urllib.request.urlopen(req).read()

    # Because the request had the Accept-Encoding: gzip header, the HTTP body was
    # compressed. Need to decompress it to access the API's response. If you prefer
    # to skip this step and rely only on snappy compression, you can remove the
    # header and use the response directly.
    decompressed_response = zlib.decompress(compressed_data,
                                            16 + zlib.MAX_WBITS)
    # print_compression_statistics('GZip', compressed_data, decompressed_response)

    # Parse the response into a Python dict
    response = json.loads(decompressed_response)
    # Extract the response, and parse back from strings to uint64
    string_values = response['uint64StrList']['values']
    values = np.array(string_values).astype(np.uint64)

    # Done!
    return values

def query_seg_ids_batch(locations, token, volume_id, query_size=128):
    all_ids = []
    for i in range(0, len(locations), query_size):
        ids = list(googlebrainmaps.query_seg_ids(np.array(locations[i:i + query_size]), token, volume_id))
        all_ids.extend(ids)
    return np.array(all_ids)