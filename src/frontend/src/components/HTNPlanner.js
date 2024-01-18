import React, { useEffect, useState } from 'react';
import io from 'socket.io-client';
import { Paper, Typography, List, ListItem, ListItemIcon, ListItemText, Collapse, IconButton } from '@mui/material';
import { ArrowRight, ExpandLess, ExpandMore } from '@mui/icons-material';

function HTNPlanner() {
  const [taskNode, setTaskNode] = useState(null);
  const [socket, setSocket] = useState(null);
  const [expandedNodes, setExpandedNodes] = useState({}); // To handle node expansion

  useEffect(() => {
    const newSocket = io('http://localhost:5000');
    setSocket(newSocket);

    newSocket.on('task_node_update', (data) => {
      setTaskNode(data);
    });

    return () => newSocket.close();
  }, []);

  const handleToggle = (node) => {
    setExpandedNodes(prev => ({
      ...prev,
      [node.node_name]: !prev[node.node_name]
    }));
  };

  const getStatusColor = (status) => {
    switch (status) {
      case "completed": return "green";
      case "in-progress": return "blue";
      default: return "grey";
    }
  };

  // Render the task node as a tree
  const renderTaskNode = (node, depth = 0) => {
    if (!node) return null;

    return (
      <List style={{ marginLeft: depth * 20 }}>
        <ListItem>
          <ListItemIcon>
            <ArrowRight style={{ color: getStatusColor(node.status) }} />
          </ListItemIcon>
          <ListItemText primary={`${node.task_name} (${node.status})`} />
          {node.children.length > 0 && (
            <IconButton edge="end" onClick={() => handleToggle(node)}>
              {expandedNodes[node.node_name] ? <ExpandLess /> : <ExpandMore />}
            </IconButton>
          )}
        </ListItem>
        <Collapse in={expandedNodes[node.node_name]} timeout="auto" unmountOnExit>
          {node.children.map((child) => renderTaskNode(child, depth + 1))}
        </Collapse>
      </List>
    );
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', marginTop: 20 }}>
      <Typography variant="h4" gutterBottom>
        HTN Planner Visualization
      </Typography>
      <Paper style={{ width: '80%', padding: 20, overflow: 'hidden' }}>
        {renderTaskNode(taskNode)}
      </Paper>
    </div>
  );
}

export default HTNPlanner;
