from gunpowder.freezable import Freezable
import logging
import numpy as np

logger = logging.getLogger(__name__)


class Points(Freezable):
    '''A list of :class:``Point``s with a specification describing the data.

    Args:

        data (dict, int->Point): A dictionary of IDs mapping to
            :class:``Point``s.

        spec (:class:`PointsSpec`): A spec describing the data.
    '''

    def __init__(self, data, spec):
        self.data = data
        self.spec = spec
        self.freeze()


class Point(Freezable):

    def __init__(self, location):
        self.location = np.array(location, dtype=np.float32)
        self.freeze()

    def __repr__(self):
        return str(self.location)


class PointsKey(Freezable):
    '''A key to identify lists of points in requests, batches, and across
    nodes.

    Used as key in :class:``BatchRequest`` and :class:``Batch`` to retrieve
    specs or lists of points.

    Args:

        identifier (string):
            A unique, human readable identifier for this points key. Will be
            used in log messages and to look up points in requests and batches.
            Should be upper case (like ``CENTER_POINTS``). The identifier is
            unique: Two points keys with the same identifier will refer to the
            same points.
    '''

    def __init__(self, identifier):
        self.identifier = identifier
        self.hash = hash(identifier)
        self.freeze()
        logger.debug("Registering points type %s", self)
        setattr(PointsKeys, self.identifier, self)

    def __eq__(self, other):
        return hasattr(other, 'identifier') and self.identifier == other.identifier

    def __hash__(self):
        return self.hash

    def __repr__(self):
        return self.identifier


class PointsKeys:
    '''Convenience access to all created :class:``PointsKey``s. A key generated
    with::

        centers = PointsKey('CENTER_POINTS')

    can be retrieved as::

        PointsKeys.CENTER_POINTS
    '''
    pass


class PreSynPoint(Point):
    def __init__(self, location, location_id, synapse_id, partner_ids, props=None):
        """ Presynaptic locations
        :param location:     ndarray, [zyx]
        :param location_id:  int, unique for every synapse location across pre and postsynaptic locations
        :param synapse_id:   int, unique for every synapse(synaptic partners have the same synapse_id, but different location_ids)
        :param partner_ids:  list of ints, location ids of postsynaptic partners
        :param props:        dict, properties

        Originally located under the gunpowder.contrib.nodes
        """
        Point.__init__(self, location=location)
        self.thaw()

        self.location_id = location_id
        self.synapse_id = synapse_id
        self.partner_ids = partner_ids
        if props is None:
            self.props = {}
        else:
            self.props = props

        self.freeze()


class PostSynPoint(Point):
    def __init__(self, location, location_id, synapse_id, partner_ids, props=None):
        """
        :param location:     ndarray, [zyx]
        :param location_id:  int, unique for every synapse location across pre and postsynaptic locations
        :param synapse_id:   int, unique for every synapse(synaptic partners have the same synapse_id, but different location_ids)
        :param partner_ids:  list of int, location id of presynaptic partner
        :param props:        dict, properties
        """
        Point.__init__(self, location=location)
        self.thaw()

        self.location_id = location_id
        self.synapse_id = synapse_id
        self.partner_ids = partner_ids
        if props is None:
            self.props = {}
        else:
            self.props = props

        self.freeze()
