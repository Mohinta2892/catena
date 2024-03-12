from gunpowder.freezable import Freezable
import logging
from gunpowder.graph import GraphKeys, Node

logger = logging.getLogger(__name__)


class PreSynPoint(Node):
    def __init__(self, location, location_id, synapse_id, partner_ids, props=None):
        """ Presynaptic locations
        :param location:     ndarray, [zyx]
        :param location_id:  int, unique for every synapse location across pre and postsynaptic locations
        :param synapse_id:   int, unique for every synapse(synaptic partners have the same synapse_id, but different location_ids)
        :param partner_ids:  list of ints, location ids of postsynaptic partners
        :param props:        dict, properties

        Originally located under the gunpowder.contrib.nodes
        """
        Node.__init__(self, id=location_id, location=location)
        self.thaw()

        # self.location_id = location_id
        self.synapse_id = synapse_id
        self.partner_ids = partner_ids
        if props is None:
            self.props = {}
        else:
            self.props = props

        self.freeze()


class PostSynPoint(Node):
    def __init__(self, location, location_id, synapse_id, partner_ids, props=None):
        """
        :param location:     ndarray, [zyx]
        :param location_id:  int, unique for every synapse location across pre and postsynaptic locations
        :param synapse_id:   int, unique for every synapse(synaptic partners have the same synapse_id, but different location_ids)
        :param partner_ids:  list of int, location id of presynaptic partner
        :param props:        dict, properties
        """
        Node.__init__(self, id=location_id, location=location)
        self.thaw()

        # self.location_id = location_id
        self.synapse_id = synapse_id
        self.partner_ids = partner_ids
        if props is None:
            self.props = {}
        else:
            self.props = props

        self.freeze()
