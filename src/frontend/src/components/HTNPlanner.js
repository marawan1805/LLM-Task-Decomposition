import React, { useEffect, useState } from 'react';
import io from 'socket.io-client';
import { Paper, Typography, List, ListItem, ListItemIcon, ListItemText, Collapse, IconButton } from '@mui/material';
import { ArrowRight, ExpandLess, ExpandMore } from '@mui/icons-material';

function HTNPlanner() {
  const [taskNode, setTaskNode] = useState(null);
  const [expandedNodes, setExpandedNodes] = useState({});

  useEffect(() => {
    const newSocket = io('http://127.0.0.1:5000');

    newSocket.on('connect', () => {
      console.log('Socket connected');
    });

    newSocket.on('task_node_update', (data) => {
      console.log('Received task_node_update:', data);
      setTaskNode(data);
      setExpandedNodes(prev => ({ ...prev, [data.task_name]: true }));
    });

    return () => newSocket.close();
  }, []);

  const handleToggle = (node) => {
    setExpandedNodes(prev => ({
      ...prev,
      [node.task_name]: !prev[node.task_name]
    }));
  };

  const getStatusColor = (status) => {
    switch (status) {
      case "completed": return "green";
      case "in-progress": return "blue";
      case "failed": return "red";
      default: return "grey";
    }
  };

  const renderTaskNode = (node, depth = 0) => {
    if (!node) return null;

    return (
      <List style={{ marginLeft: depth * 20 }}>
        <ListItem>
          <ListItemIcon>
            <ArrowRight style={{ color: getStatusColor(node.status) }} />
          </ListItemIcon>
          <ListItemText primary={`${node.task_name} (${node.status})`} />
          {node.children && node.children.length > 0 && (
            <IconButton edge="end" onClick={() => handleToggle(node)}>
              {expandedNodes[node.task_name] ? <ExpandLess /> : <ExpandMore />}
            </IconButton>
          )}
        </ListItem>
        <Collapse in={expandedNodes[node.task_name]} timeout="auto" unmountOnExit>
          {node.children && node.children.map((child, index) => renderTaskNode(child, depth + 1))}
        </Collapse>
      </List>
    );
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', marginTop: 20 }}>
      <Typography variant="h4" gutterBottom>
        HTN Planner Visualization
      </Typography>
      <Paper style={{ width: '80%', padding: 20, overflow: 'auto', maxHeight: '80vh' }}>
        {taskNode ? renderTaskNode(taskNode) : <Typography>Waiting for data...</Typography>}
      </Paper>
    </div>
  );
}

export default HTNPlanner;